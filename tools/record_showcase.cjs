/* Optional recording utility: npm install playwright, then serve repo on 8768. */
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('fs');
const path=require('path');
(async()=>{
 const output=path.resolve(process.env.DEMO_OUTPUT||'docs/media');fs.mkdirSync(output,{recursive:true});
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--no-proxy-server']});
 const context=await browser.newContext({viewport:{width:1440,height:1000},recordVideo:{dir:path.resolve('demo/recordings'),size:{width:1440,height:1000}}});
 const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DEMO_URL||'http://127.0.0.1:8768/demo/');await page.waitForSelector('#run-rows tr');
 await page.addStyleTag({content:'#demo-caption{position:fixed;bottom:22px;left:220px;right:24px;z-index:99;background:#142033;color:#fff;padding:18px 26px;border:1px solid #ffffff33;border-radius:8px;font:600 30px/1.55 "Microsoft YaHei",sans-serif;box-shadow:0 4px 24px #0005;pointer-events:none;text-wrap:pretty} .panel{font-size:16px} .hint,.muted,.choice small{font-size:14px} .chart-caption{font-size:14px}'});
 await page.evaluate(()=>{const d=document.createElement('div');d.id='demo-caption';document.body.append(d);});
 async function caption(text,seconds=5){await page.locator('#demo-caption').evaluate((el,t)=>{el.textContent=t},text);await page.waitForTimeout(seconds*1000);}
 await caption('EmbodiQ：具身模型量化评测平台。对比任务成功率、响应速度和显存占用。',5);
 await page.locator('nav a[href="#create"]').click();await caption('第一步：选择模型。支持 OpenVLA、VLA-Adapter、FastWAM，可同时评测多个模型。',5);
 await caption('第二步：选择原始精度与四位量化，在相同任务中比较模型表现。',5);
 await page.waitForFunction(()=>document.getElementById('estimate').textContent.includes('12,000'));
 await caption('评测覆盖空间关系、目标物体、任务目标和长时程操作，共 12,000 回合。',7);
 await page.locator('nav a[href="#tasks"]').click();await page.evaluate(()=>scrollTo(0,0));await caption('任务管理：按模型、量化方法或运行状态查找评测记录。',6);
 await page.locator('#monitor-title').scrollIntoViewIfNeeded();await page.evaluate(()=>scrollBy(0,170));await caption('查看实验详情，核对模型、方法与任务集合。这里展示的是已完成的评测结果。',6);
 await page.locator('nav a[href="#results"]').click();await page.evaluate(()=>scrollTo(0,0));await page.getByRole('button',{name:'读取结果',exact:true}).click();await page.waitForSelector('.chart-grid');
 await caption('结果对比：观察量化前后的成功率、响应时间与显存占用，判断性能与资源之间的取舍。',7);
 await page.locator('#demo-caption').evaluate(el=>el.hidden=true);await page.screenshot({path:path.join(output,'results.png'),fullPage:true});await page.locator('#demo-caption').evaluate(el=>el.hidden=false);
 await page.getByRole('button',{name:'空间关系',exact:true}).click();await caption('切换任务集合，查看模型在不同操作任务中的表现。',5);
 await page.locator('#report table').scrollIntoViewIfNeeded();await caption('查看分项数据，也可以下载报告与数据表，用于汇报和分析。',6);
 await page.locator('.glossary summary').click();await page.locator('.glossary').scrollIntoViewIfNeeded();await caption('展开中文指标说明，了解各项数值的含义与比较方式。',6);
 await page.locator('nav a[href="#tasks"]').click();await page.evaluate(()=>scrollTo(0,0));await caption('选择模型与量化方法，统一评测、管理任务、对比结果。使用方法见项目首页。',5);
 await page.locator('#demo-caption').evaluate(el=>el.remove());await page.screenshot({path:path.join(output,'tasks.png'),fullPage:true});
 await page.locator('nav a[href="#create"]').click();await page.screenshot({path:path.join(output,'configuration.png'),fullPage:true});
 const video=page.video();await context.close();await video.saveAs(path.join(output,'embodiq-demo.webm'));await browser.close();
 fs.writeFileSync(path.join(output,'recording.json'),JSON.stringify({page_errors:errors,source:'demo/index.html',read_only:true,real_evaluation_submitted:false,format:'webm',viewport:[1440,1000],caption_font_px:30},null,2));
 if(errors.length)throw Error(errors.join('\n'));console.log('recording complete: '+output);
})().catch(e=>{console.error(e);process.exit(1)});
