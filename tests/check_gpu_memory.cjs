/* Optional browser check: PLAYWRIGHT_MODULE=/path/to/playwright node tests/check_gpu_memory.cjs */
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'..'),staticRoot=path.join(root,'src/embodiq/web/static');
const out=fs.mkdtempSync(path.join(os.tmpdir(),'embodiq-memory-'));
const snapshot=JSON.parse(fs.readFileSync(path.join(root,'demo/demo-data.js'),'utf8').replace(/^window\.EMBODIQ_DEMO\s*=\s*/,'').replace(/;\s*$/,''));
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[],calls=[],submissions=[],previews=[];
  page.on('pageerror',error=>errors.push(error.message));
  const readings={0:{total_gib:24,used_gib:8,free_gib:15.5},2:{total_gib:24,used_gib:14,free_gib:9},5:{total_gib:16,used_gib:12,free_gib:3.75}};
  let releaseOld,sawOld,fail=false,devices=[0,2,5],listFail=false;
  const delayed=new Promise(resolve=>{releaseOld=resolve;});
  const oldStarted=new Promise(resolve=>{sawOld=resolve;});
  await page.route('http://embodiq.test/**',async route=>{
   const url=new URL(route.request().url()),json=value=>route.fulfill({contentType:'application/json',body:JSON.stringify(value)});
   if(url.pathname==='/api/capabilities')return json(snapshot.capabilities);
   if(url.pathname==='/api/gpus')return json(listFail?{status:'unavailable',gpus:[]}:{status:'ok',gpus:devices});
   if(url.pathname==='/api/runs'){
    if(route.request().method()==='POST'){
     submissions.push(route.request().postDataJSON());
     return route.fulfill({status:400,contentType:'application/json',body:JSON.stringify({error:'测试已拦截提交，不启动模型。'})});
    }
    return json({runs:[],errors:[]});
   }
   if(url.pathname==='/api/preview'){previews.push(route.request().postDataJSON());return json({episodes:12000,total_stages:31});}
   if(url.pathname==='/api/gpu-memory'){
    const gpu=Number(url.searchParams.get('gpu'));calls.push(gpu);
    if(gpu===2){sawOld();await delayed;}
    if(fail)return route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:'private device error'})});
    return json({status:'ok',memory:readings[gpu],sampled_at:'2026-10-12T06:30:00+00:00',message:''});
   }
   const name=url.pathname==='/'?'index.html':url.pathname.slice(1);
   if(!['index.html','app.js','style.css'].includes(name))return route.fulfill({status:404,body:''});
   return route.fulfill({contentType:{'index.html':'text/html','app.js':'text/javascript','style.css':'text/css'}[name],body:fs.readFileSync(path.join(staticRoot,name))});
  });
  await page.goto('http://embodiq.test/#create');
  await page.waitForFunction(()=>document.getElementById('gpu-memory-free').textContent==='15.50GiB');
  assert.deepEqual(await page.locator('#gpu option').evaluateAll(options=>options.map(o=>o.value)),['0','2','5']);
  assert((await page.locator('#gpu-list-status').innerText()).includes('3 张显卡'));
  assert.equal(await page.locator('#gpu-memory-meter').evaluate(el=>el.value),1/3);
  assert(!(await page.locator('.gpu-memory').innerText()).match(/NVIDIA|RTX|CUDA|UUID|驱动|型号|版本|设备编号/));
  await page.locator('#gpu').selectOption('2');await oldStarted;
  await page.waitForFunction(()=>document.getElementById('gpu-memory-values').hidden);
  await page.locator('#gpu').selectOption('5');
  await page.waitForFunction(()=>document.getElementById('gpu-memory-free').textContent==='3.75GiB');
  releaseOld();await page.waitForTimeout(100);
  assert.equal(await page.locator('#gpu-memory-free').textContent(),'3.75GiB','过期的显卡请求不得覆盖当前选择');
  await page.getByRole('button',{name:'刷新列表',exact:true}).click();
  await page.waitForFunction(()=>!document.getElementById('gpu').disabled);
  assert.equal(await page.locator('#gpu').inputValue(),'5','刷新列表应保留当前选择');
  await page.waitForFunction(()=>!document.getElementById('submit').disabled);
  await page.locator('#submit').click();
  await page.waitForFunction(()=>document.getElementById('notice').textContent.includes('测试已拦截'));
  assert.equal(submissions.length,1);assert.equal(submissions[0].gpu,5);
  assert(previews.some(body=>body.gpu===5),'预览计划应使用所选显卡编号');
  readings[5].free_gib=3.5;
  await page.getByRole('button',{name:'刷新显存',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('gpu-memory-free').textContent==='3.50GiB');
  readings[5].free_gib=3.25;
  await page.waitForFunction(()=>document.getElementById('gpu-memory-free').textContent==='3.25GiB',{},{timeout:5500});
  fail=true;
  await page.getByRole('button',{name:'刷新显存',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('gpu-memory-status').textContent.includes('暂无法读取'));
  assert(await page.locator('#gpu-memory-values').isHidden());
  assert.equal(await page.locator('#gpu-memory-free').textContent(),'—');
  assert(!(await page.locator('.gpu-memory').innerText()).includes('private device error'));
  fail=false;devices=[0,2];
  await page.getByRole('button',{name:'刷新列表',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('gpu-list-status').textContent.includes('重新选择'));
  assert.equal(await page.locator('#gpu').inputValue(),'','设备消失时不得自动切换到其他显卡');
  assert(await page.locator('#submit').isDisabled());
  assert((await page.locator('#gpu-memory-status').innerText()).includes('选择显卡后'));
  await page.locator('#gpu').selectOption('0');
  await page.waitForFunction(()=>document.getElementById('gpu-memory-free').textContent==='15.50GiB');
  assert((await page.locator('#gpu-list-status').innerText()).includes('2 张显卡'),'重新选择后应清除旧提示');
  for(const scheme of ['light','dark']){
   await page.emulateMedia({colorScheme:scheme});
   for(const width of [390,1440]){
    await page.setViewportSize({width,height:1100});await page.locator('.gpu-memory').scrollIntoViewIfNeeded();
    const overflows=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
    assert(!overflows,`${scheme} ${width}: 页面溢出`);
    await page.screenshot({path:path.join(out,`${scheme}-${width}.png`)});
   }
  }
  listFail=true;
  await page.getByRole('button',{name:'刷新列表',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('gpu-list-status').textContent.includes('暂无法读取'));
  assert(await page.locator('#gpu').isDisabled());assert(await page.locator('#submit').isDisabled());
  assert(await page.locator('#gpu-memory-values').isHidden());
  await page.locator('#create-form').evaluate(form=>form.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})));
  assert.equal(submissions.length,1,'无法读取列表时不得把空值当作显卡 0 提交');
  listFail=false;devices=[];
  await page.getByRole('button',{name:'刷新列表',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('gpu-list-status').textContent.includes('未检测到'));
  assert(await page.locator('#gpu').isDisabled());assert(await page.locator('#submit').isDisabled());
  assert(!previews.some(body=>body.gpu===null));
  // Loading from another tab still prepares the selector for resume/new runs.
  devices=[2,5];
  await page.goto('http://embodiq.test/#tasks');
  await page.reload();
  await page.waitForFunction(()=>document.getElementById('gpu').value==='2');
  await page.getByRole('link',{name:'新建评测',exact:true}).first().click();
  await page.waitForFunction(()=>document.getElementById('gpu-memory-free').textContent==='9.00GiB');
  // The offline showcase must not present fixtures as live hardware readings.
  await page.goto(pathToFileURL(path.join(root,'demo/index.html')).href+'#create');
  await page.waitForFunction(()=>document.getElementById('gpu-memory-status').textContent.includes('只读演示'));
  assert((await page.locator('#gpu').innerText()).includes('演示'));
  assert(await page.locator('#gpu-memory-values').isHidden());
  assert.deepEqual(errors,[]);assert(calls.includes(0)&&calls.includes(2)&&calls.includes(5));
  console.log('PASS: 显卡枚举与实际编号提交、刷新保留选择、设备消失/读取失败/无卡状态、显存切换与刷新、浅深主题与手机布局、演示无实时数据。');
  console.log('截图：'+out);
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
