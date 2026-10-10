/* Optional tool: PLAYWRIGHT_MODULE=... DEMO_URL=... node tools/record_showcase.cjs.
 * Record a fixed viewport only. Capture README screenshots in a separate context.
 */
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('fs'),path=require('path');
const viewport={width:1600,height:900};
(async()=>{
 const output=path.resolve(process.env.DEMO_OUTPUT||'docs/media');fs.mkdirSync(output,{recursive:true});
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const context=await browser.newContext({viewport,deviceScaleFactor:1,recordVideo:{dir:path.resolve('demo/recordings'),size:viewport}});
 const page=await context.newPage(),errors=[],scenes=[],checks=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL||'http://127.0.0.1:8768/demo/');
 await page.waitForFunction(()=>document.getElementById('estimate').textContent.includes('12,000'));
 await page.addStyleTag({content:`
  body{padding-bottom:145px}html{scroll-behavior:auto!important}
  #record-caption{position:fixed;inset:auto 0 0;height:114px;z-index:1000;background:#132c27;color:#fff;display:flex;align-items:center;gap:30px;padding:22px 42px;border-top:3px solid #658c77;font-family:"Microsoft YaHei",sans-serif;pointer-events:none}
  #record-chapter{flex:0 0 178px;color:#aec8ba;font-size:16px;line-height:1.8;border-right:1px solid #496157;padding-right:22px}
  #record-chapter strong{display:block;color:#fff;font-size:23px;font-weight:600}
  #record-text{font-size:29px;line-height:1.5;max-width:1230px;font-weight:550}
  #record-pointer{position:fixed;z-index:1001;width:24px;height:28px;pointer-events:none;left:0;top:0;transition:transform 400ms ease;filter:drop-shadow(0 1px 2px #0004)}
  #record-pointer.pressed{filter:drop-shadow(0 0 5px #176e63)}
  .panel{font-size:16px}.hint,.muted,.choice small{font-size:14px}.chart-caption{font-size:14px}
 `});
 await page.evaluate(()=>{
  const caption=document.createElement('div');caption.id='record-caption';caption.innerHTML='<div id="record-chapter"></div><div id="record-text"></div>';document.body.append(caption);
  const pointer=document.createElement('div');pointer.id='record-pointer';pointer.style.transform='translate(1500px,70px)';pointer.innerHTML='<svg viewBox="0 0 24 28"><path d="M2 1L3 23L9 17L13 26L17 24L13 16L22 15Z" fill="white" stroke="#18332e" stroke-width="1.5"/></svg>';document.body.append(pointer);
  window.recordViewportViolations=[];
  window.recordViewportTimer=setInterval(()=>{if(innerWidth!==1600||innerHeight!==900)window.recordViewportViolations.push([performance.now(),innerWidth,innerHeight]);},40);
 });
 const start=Date.now();
 const hold=seconds=>page.waitForTimeout(seconds*1000);
 async function settle(){await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));}
 async function focus(selector,offset=90){
  await page.locator(selector).first().evaluate((el,y)=>window.scrollTo({top:Math.max(0,window.scrollY+el.getBoundingClientRect().top-y),behavior:'smooth'}),offset);
  await hold(.9);await settle();
 }
 async function point(selector){
  const box=await page.locator(selector).first().boundingBox();if(!box)throw Error('Missing pointer target: '+selector);
  if(box.y<0||box.y+box.height>viewport.height-118)await focus(selector,260);
  const final=await page.locator(selector).first().boundingBox(),x=final.x+Math.min(final.width/2,120),y=final.y+final.height/2;
  await page.locator('#record-pointer').evaluate((el,position)=>{el.style.transform=`translate(${position.x}px,${position.y}px)`;}, {x,y});
  await page.mouse.move(x,y);await hold(.65);
 }
 async function click(selector){
  await point(selector);await page.locator('#record-pointer').evaluate(el=>el.classList.add('pressed'));
  await page.locator(selector).first().click();await hold(.45);await page.locator('#record-pointer').evaluate(el=>el.classList.remove('pressed'));await settle();
 }
 async function scene(chapter,title,text,action,seconds=6){
  const item={chapter,title,text,start_s:(Date.now()-start)/1000};
  await page.evaluate(({chapter,title,text})=>{document.getElementById('record-chapter').replaceChildren(document.createTextNode(chapter+' / 05'));const strong=document.createElement('strong');strong.textContent=title;document.getElementById('record-chapter').append(strong);document.getElementById('record-text').textContent=text;},{chapter,title,text});
  await hold(1);if(action)await action();await settle();
  const dimensions=await page.evaluate(()=>({width:innerWidth,height:innerHeight,captionHeight:document.getElementById('record-caption').getBoundingClientRect().height}));
  if(dimensions.width!==1600||dimensions.height!==900||dimensions.captionHeight!==114)throw Error('Viewport changed during recording');
  checks.push(dimensions);item.settled_s=(Date.now()-start)/1000;
  await hold(seconds);item.end_s=(Date.now()-start)/1000;scenes.push(item);
 }
 await scene('01','选择配置','从选择模型到导出报告，用同一组配置走完整个评测流程。',null,6);
 await scene('01','选择模型','以 VLA-Adapter 为例，对比原始精度与 RTN 四位量化。',async()=>{
  await click('#models label:has(input[value="openvla"])');await hold(1);
  await click('#models label:has(input[value="fastwam"])');
  await page.waitForFunction(()=>document.getElementById('estimate').textContent.includes('4,000'));
 },7);
 await scene('01','任务范围','保留四类任务，右侧汇总所选模型、方法和评测规模。',async()=>{await point('#suites');},6);
 await scene('02','查看任务','点击“查看本配置示例”，进入对应的任务记录。',async()=>{
  await click('#submit');await page.locator('#view-tasks').waitFor({state:'visible'});await hold(.5);
 },6);
 await scene('02','核对内容','任务评测、资源测量和报告，均对应刚才选定的配置。',async()=>{await focus('#demo-coverage',230);},7);
 await scene('03','任务表现','查看成功率折线图：灰色为原始精度，绿色为 RTN。',async()=>{
  await click('.demo-next a');await page.waitForSelector('.trend-grid');await focus('.chart-heading',100);
 },9);
 await scene('03','读取图表','横轴为四类任务，纵轴为成功率；每个点标出对应数值。',async()=>{await point('.trend-svg [data-method="rtn"][data-suite="libero_goal"]');},7);
 await scene('04','速度与显存','切换到资源对比，同时观察查询耗时和显存变化。',async()=>{
  await click('.chart-mode button:has-text("速度与显存")');await page.waitForSelector('.chart-grid.resources');await focus('.chart-heading',110);
 },9);
 await scene('04','按任务查看','也可以切换任务集合，比较同一类任务下的资源占用。',async()=>{
  await focus('.report-toolbar',140);await click('.suite-tabs button:has-text("空间关系")');await focus('.chart-heading',110);
 },7);
 await scene('05','导出报告','回到全部集合，下载报告和数据表，保留完整对比结果。',async()=>{
  await focus('.report-toolbar',140);await click('.suite-tabs button:has-text("全部集合")');
  await focus('.downloads',250);const download=page.waitForEvent('download');await click('.downloads a:has-text("下载完整报告")');await download;
 },7);
 await scene('05','导出图表','任务成功率图支持 SVG 导出，可直接用于汇报或论文排版。',async()=>{
  const download=page.waitForEvent('download');await click('.downloads button:has-text("下载任务对比图")');await download;
 },7);
 await scene('05','完成','配置评测 → 查看任务 → 对比结果。演示不启动模型。',async()=>{
  await focus('.chart-heading',100);await click('.chart-mode button:has-text("任务表现")');await page.waitForSelector('.trend-grid');await focus('.chart-heading',100);
 },6);
 const violations=await page.evaluate(()=>{clearInterval(window.recordViewportTimer);return window.recordViewportViolations;});
 const video=page.video();await context.close();await video.saveAs(path.join(output,'embodiq-demo.webm'));await browser.close();
 fs.writeFileSync(path.join(output,'recording.json'),JSON.stringify({page_errors:errors,source:'demo/index.html',read_only:true,real_evaluation_submitted:false,format:'webm',viewport,caption_font_px:29,caption_reserved_height:114,viewport_checks:checks,viewport_violations:violations,scenes,screenshots_during_recording:0},null,2));
 if(errors.length||violations.length)throw Error(JSON.stringify({errors,violations}));
 console.log(`recording complete: ${scenes.length} scenes, fixed ${viewport.width} × ${viewport.height}`);
})().catch(e=>{console.error(e);process.exit(1)});
