/* Capture documentation stills separately from the fixed-viewport recording. */
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const path=require('path');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1600,height:1100}});
 await page.goto(process.env.DEMO_URL||'http://127.0.0.1:8768/demo/');
 await page.waitForFunction(()=>!document.getElementById('submit').disabled);
 for(const [view,file] of [['create','configuration'],['tasks','tasks'],['results','results']]){
  await page.locator(`nav a[href="#${view}"]`).click();await page.locator(`#view-${view}`).waitFor({state:'visible'});
  if(view==='results'){
   await page.waitForSelector('.trend-grid');
   await page.locator('.report-toolbar').evaluate(el=>window.scrollTo(0,window.scrollY+el.getBoundingClientRect().top-100));
  }
  await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
  await page.screenshot({path:path.join(process.env.DEMO_OUTPUT||'docs/media',file+'.png'),fullPage:view!=='results'});
 }
 await page.getByRole('button',{name:'速度与显存',exact:true}).click();
 await page.waitForSelector('.chart-grid.resources');
 await page.locator('.report-toolbar').evaluate(el=>window.scrollTo(0,window.scrollY+el.getBoundingClientRect().top-100));
 await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
 await page.screenshot({path:path.join(process.env.DEMO_OUTPUT||'docs/media','resources.png')});
 await browser.close();console.log('documentation screenshots complete (unrecorded context)');
})().catch(e=>{console.error(e);process.exit(1)});
