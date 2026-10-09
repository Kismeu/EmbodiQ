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
 await page.addStyleTag({content:'#demo-caption{position:fixed;bottom:22px;left:220px;right:24px;z-index:99;background:#202b40;color:#fff;padding:15px 22px;border-radius:6px;font:16px/1.7 "Microsoft YaHei",sans-serif;box-shadow:0 4px 24px #0002;pointer-events:none}'});
 await page.evaluate(()=>{const d=document.createElement('div');d.id='demo-caption';document.body.append(d);});
 async function caption(text,seconds=5){await page.locator('#demo-caption').evaluate((el,t)=>{el.textContent=t},text);await page.waitForTimeout(seconds*1000);}
 await caption('EmbodiQ：具身模型量化评测工作台。本视频展示真实快速检查记录，配置预览不会启动模型。',5);
 await page.locator('nav a[href="#create"]').click();await caption('第一步：选择模型。支持 OpenVLA、VLA-Adapter、FastWAM，可同时评测多个模型。',5);
 await caption('第二步：选择原始精度或四位量化。AWQ、HSA 暂未接入，界面中保留为不可选状态。',5);
 await page.locator('#profile').selectOption('formal');for(const value of ['libero_object','libero_goal','libero_10'])await page.locator(`input[value="${value}"]`).check();
 await page.waitForFunction(()=>document.getElementById('estimate').textContent.includes('12,000'));
 await caption('正式评测覆盖四类机器人操作任务。三模型 × 两方法 × 四集合，共 12,000 回合、31 个执行阶段。',7);
 await page.locator('nav a[href="#tasks"]').click();await page.evaluate(()=>scrollTo(0,0));await caption('任务管理：按模型或运行状态查找实验。这里展示的是已完成的快速检查，不是实时重跑。',6);
 await page.locator('#monitor-title').scrollIntoViewIfNeeded();await page.evaluate(()=>scrollBy(0,170));await caption('每次评测依次完成机器人任务、速度与显存测量、报告生成。进度和日志都可追溯。',6);
 await page.locator('nav a[href="#results"]').click();await page.evaluate(()=>scrollTo(0,0));await page.getByRole('button',{name:'读取结果',exact:true}).click();await page.waitForSelector('.chart-grid');
 await caption('结果对比：成功率越高越好，响应时间与显存占用越低越好。当前样本仅用于验证流程。',7);
 await page.locator('#demo-caption').evaluate(el=>el.hidden=true);await page.screenshot({path:path.join(output,'results.png'),fullPage:true});await page.locator('#demo-caption').evaluate(el=>el.hidden=false);
 await page.getByRole('button',{name:'空间关系',exact:true}).click();await caption('按任务集合查看细分结果，避免把不同能力混为一个分数。',5);
 await page.locator('#report table').scrollIntoViewIfNeeded();await caption('表格同时保留任务成功率、回合数、响应时间，以及运行与加载两个阶段的显存峰值。',6);
 await page.locator('.glossary summary').click();await page.locator('.glossary').scrollIntoViewIfNeeded();await caption('不熟悉指标时，可展开中文说明。完整报告和数据表可以下载，用于复核与后续分析。',6);
 await page.locator('nav a[href="#tasks"]').click();await page.evaluate(()=>scrollTo(0,0));await caption('源码、本地工作台与只读演示一并提供。正式测评需要另行准备模型、仿真环境与固定评测输入。',5);
 await page.locator('#demo-caption').evaluate(el=>el.remove());await page.screenshot({path:path.join(output,'tasks.png'),fullPage:true});
 await page.locator('nav a[href="#create"]').click();await page.screenshot({path:path.join(output,'configuration.png'),fullPage:true});
 const video=page.video();await context.close();await video.saveAs(path.join(output,'embodiq-demo.webm'));await browser.close();
 fs.writeFileSync(path.join(output,'recording.json'),JSON.stringify({page_errors:errors,source:'demo/index.html',read_only:true,real_evaluation_submitted:false,format:'webm',viewport:[1440,1000]},null,2));
 if(errors.length)throw Error(errors.join('\n'));console.log('recording complete: '+output);
})().catch(e=>{console.error(e);process.exit(1)});
