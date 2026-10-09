/* Read-only local transport for the showcase. No model execution or network API. */
(() => {
  'use strict';
  const data=window.EMBODIQ_DEMO;
  const json=(value,status=200)=>Promise.resolve(new Response(JSON.stringify(value),{status,headers:{'Content-Type':'application/json'}}));
  window.fetch=async(input,options={})=>{
    const path=String(input).split('?')[0];
    if(path==='/api/capabilities')return json(data.capabilities);
    if(path==='/api/preview'){
      const request=JSON.parse(options.body||'{}');
      for(const [key,allowed] of [['models',data.capabilities.models],['methods',['fp16','rtn']],['suites',data.capabilities.suites]]){
        if(!Array.isArray(request[key])||!request[key].length||request[key].some(x=>!allowed.includes(x)))return json({error:'请选择支持的模型、方法和任务集合。'},400);
      }
      const combinations=request.models.length*request.methods.length,rollouts=combinations*request.suites.length;
      return json({total_stages:rollouts+combinations+1,episodes:rollouts*(request.profile==='formal'?500:1),profile:request.profile});
    }
    if(path==='/api/runs'&&!options.method)return json({runs:data.runs,errors:[]});
    const match=path.match(/^\/api\/runs\/([^/]+)(?:\/(report|logs))?$/);
    if(match&&!options.method){
      if(match[2]==='report')return data.reports[match[1]]?json(data.reports[match[1]]):json({error:'演示中没有该报告。'},404);
      if(match[2]==='logs')return json({text:'只读演示不导出原始日志。真实运行中可在这里查看模型加载、评测阶段和错误信息。',truncated:false});
      return json(data.runs.find(run=>run.run_id===match[1])||{});
    }
    return json({error:'这是只读演示，不会启动或修改实验。请安装本地工作台后执行评测。'},403);
  };
  // Report export remains usable offline; no original local file paths included.
  document.addEventListener('click',event=>{
    const link=event.target.closest('a');const match=link?.getAttribute('href')?.match(/^\/api\/runs\/([^/]+)\/report\/(report\.md|cells\.csv|tradeoffs\.svg)$/);
    if(!match)return;event.preventDefault();const report=data.reports[match[1]];if(!report)return;
    const file=match[2];let text,mime;
    if(file==='tradeoffs.svg'){
      document.getElementById('notice').hidden=false;document.getElementById('notice').textContent='演示版提供报告与数据表下载。完整对比图请在本地评测工作台导出。';return;
    }
    if(file==='cells.csv'){
      const fields=Object.keys(report.cells[0]);const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';
      text='\uFEFF'+[fields,...report.cells.map(row=>fields.map(k=>row[k]))].map(row=>row.map(quote).join(',')).join('\n');mime='text/csv;charset=utf-8';
    }else{
      text='# EmbodiQ 只读演示报告\n\n真实快速检查记录，不代表正式任务性能。\n\n'+JSON.stringify(report,null,2);mime='text/plain;charset=utf-8';
    }
    const url=URL.createObjectURL(new Blob([text],{type:mime})),a=document.createElement('a');a.href=url;a.download=file;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
})();
