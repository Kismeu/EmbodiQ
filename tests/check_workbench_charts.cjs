/* Optional browser regression: PLAYWRIGHT_MODULE=/path/to/playwright node tests/check_workbench_charts.cjs */
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'..'),out=fs.mkdtempSync(path.join(os.tmpdir(),'embodiq-charts-'));
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  assert.equal(fs.readFileSync(path.join(root,'demo/app.js'),'utf8'),fs.readFileSync(path.join(root,'src/embodiq/web/static/app.js'),'utf8'));
  const page=await browser.newPage({viewport:{width:1600,height:1100}}),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto(pathToFileURL(path.join(root,'demo/index.html')).href+'#results');
  await page.waitForSelector('.trend-grid');
  const snapshot=await page.evaluate(()=>JSON.parse(JSON.stringify(reportData)));
  assert(!/\bpp\b/.test(await page.locator('#report').innerText()));
  assert((await page.locator('.comparison-conclusions').innerText()).includes('76.95% 降至 73.95%（−3.00%）'));
  assert((await page.locator('.comparison-conclusions').innerText()).includes('96.80% 升至 96.85%（+0.05%）'));
  await page.getByRole('button',{name:'性能与资源权衡',exact:true}).click();
  async function checkPoints(rows,key,successKey){
   const points=await page.locator('.tradeoff-point').evaluateAll(nodes=>nodes.map(el=>({...el.dataset,cx:+el.lastElementChild.getAttribute('cx'),cy:+el.lastElementChild.getAttribute('cy')})));
   assert.equal(points.length,rows.length);
   for(const point of points){
    const row=rows.find(r=>r.model===point.model&&r.method===point.method);assert(row);
    assert.equal(+point.resource,row[key]);assert.equal(+point.success,row[successKey]);
   }
   for(const a of points)for(const b of points){
    if(+a.resource<+b.resource)assert(a.cx<b.cx,'资源轴应随数据增大');
    if(+a.success<+b.success)assert(a.cy>b.cy,'成功率越高，图中位置越高');
   }
  }
  await checkPoints(snapshot.aggregates,'steady_nvml_gib','selected_suites_success_rate');
  assert.equal(await page.locator('.tradeoff-arrow').count(),3);
  await page.locator('.tradeoff-point').first().focus();
  assert(await page.locator('.viz-tip').isVisible());assert((await page.locator('.viz-tip').innerText()).includes('GiB'));
  await page.getByRole('button',{name:'任务目标',exact:true}).click();
  const selected=snapshot.cells.filter(c=>c.suite==='libero_goal');
  await checkPoints(selected,'steady_nvml_gib','success_rate');
  const conclusions=await page.locator('.comparison-conclusions').innerText();
  assert(conclusions.includes('97.40% 降至 87.60%（−9.80%）'));assert(!conclusions.includes('−3.80%'));
  await page.getByRole('button',{name:'查询耗时',exact:true}).click();
  await checkPoints(selected,'query_ms','success_rate');
  async function exportChart(button,filename){
   const pending=page.waitForEvent('download');await page.getByRole('button',{name:button,exact:true}).click();
   const download=await pending;assert.equal(download.suggestedFilename(),filename);const target=path.join(out,filename);await download.saveAs(target);
   const svg=fs.readFileSync(target,'utf8');assert(!/NaN|Infinity|\bpp\b/.test(svg));
   assert(!await page.evaluate(text=>new DOMParser().parseFromString(text,'image/svg+xml').querySelector('parsererror')?.textContent,svg));
   return svg;
  }
  const exported=await exportChart('下载当前权衡图（SVG）','tradeoff-latency.svg');
  assert(exported.includes('任务目标'));assert(exported.includes('平均查询耗时'));
  await page.getByRole('button',{name:'峰值显存',exact:true}).click();
  await exportChart('下载当前权衡图（SVG）','tradeoff-memory.svg');
  await exportChart('下载成功率对比图（SVG）','success-rates.svg');
  await page.getByRole('button',{name:'全部集合',exact:true}).click();
  for(const theme of ['light','dark']){
   await page.emulateMedia({colorScheme:theme,reducedMotion:'reduce'});
   for(const width of [390,768,1600]){
    await page.setViewportSize({width,height:1100});
    await page.waitForTimeout(200);
    const geometry=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,width:innerWidth,
     points:[...document.querySelectorAll('.tradeoff-point circle:last-child')].map(p=>({x:+p.getAttribute('cx'),y:+p.getAttribute('cy')})),
     view:document.querySelector('.tradeoff-svg').viewBox.baseVal.width}));
    assert(geometry.scroll<=geometry.width+1,`${theme} ${width}: 页面溢出`);
    assert(geometry.points.every(p=>p.x>0&&p.x<geometry.view&&p.y>=0&&p.y<=380));
    await page.locator('.chart-heading').evaluate(el=>el.scrollIntoView({block:'start'}));
    await page.screenshot({path:path.join(out,`${theme}-${width}.png`)});
   }
  }
  // Missing counterparts and missing measurements must not become zero-valued results or claimed savings.
  await page.evaluate(doc=>{reportData=doc;reportData.aggregates=doc.aggregates.filter(r=>r.method==='rtn');reportData.cells=doc.cells.filter(r=>r.method==='rtn');renderReport();},structuredClone(snapshot));
  assert.equal(await page.locator('.tradeoff-point').count(),3);assert.equal(await page.locator('.tradeoff-arrow').count(),0);
  assert((await page.locator('.comparison-conclusions').innerText()).includes('缺少配对结果'));
  assert(!(await page.locator('.comparison-conclusions').innerText()).includes('减少'));
  await page.evaluate(doc=>{reportData=doc;for(const r of reportData.aggregates){r.steady_nvml_gib=null;r.query_ms=null;r.selected_suites_success_rate=null;}renderReport();},structuredClone(snapshot));
  assert.equal(await page.locator('.tradeoff-point').count(),0);
  assert(await page.getByRole('button',{name:'下载当前权衡图（SVG）',exact:true}).isDisabled());
  assert((await page.locator('.comparison-conclusions').innerText()).includes('可对比的指标不足'));
  await page.evaluate(doc=>{reportData=doc;for(const r of reportData.aggregates){r.steady_nvml_gib=0;r.query_ms=0;r.selected_suites_success_rate=0;}renderReport();},structuredClone(snapshot));
  assert.equal(await page.locator('.tradeoff-point').count(),6);
  assert(!/NaN|Infinity/.test(await page.locator('#report').innerHTML()));
  assert.equal(await page.locator('.tradeoff-arrow').count(),0);
  assert.deepEqual(errors,[]);
  console.log('PASS: 数据与坐标、筛选联动、中文结论、键盘提示、SVG 导出、浅深主题与三种屏宽、缺失值和零值。');
  console.log('截图与导出文件：'+out);
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
