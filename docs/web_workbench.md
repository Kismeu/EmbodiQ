# 本地 Web/API 评测工作台

2026-10-09：在已验证的 CLI 自动流程之上增加网页入口。模型、量化、LIBERO 协议、资源专测和报告准入仍由同一套 pipeline 执行。Web 服务使用 Python 标准库与本地静态页面，没有新增运行依赖。

## 启动与使用

在仓库根目录执行：

```bash
PYTHONPATH=src /data/jyc/tools/anaconda3/envs/openvla/bin/python -m embodiq.ctl.cli web \
  --config configs/stage_fp16_rtn.json --runs-root results/platform --port 8765
```

打开终端显示的 `http://127.0.0.1:8765`。端口已占用时改用其他端口，或 `--port 0` 自动选择空闲端口。服务只绑定本机；远程工作站可使用 SSH 本地端口转发，浏览器访问与服务相同的本地端口。

1. 进入“新建评测”，勾选已注册模型、可用方法和 LIBERO 集合。
2. 选择快速检查（原“冒烟验证”）或正式评测，填写显卡物理编号及忙碌时最大等待秒数。页面预览阶段数、总回合数；正式三模型 × 两方法 × 四集合为 31 阶段、12,000 回合。
3. 点击“开始评测”，自动转到“任务管理”。页面每 3 秒读取真实状态；实验记录支持搜索和状态筛选，详情中可展开运行日志、切换来源。
4. 需要中止时点击“停止评测”。等待控制器处理并释放锁后，可点击“继续评测”；续跑采用“新建评测”页当前的显卡与等待设置。
5. 进入“评测结果”，选择已完成实验并点击“读取结果”，查看分集合成功率、所选集合平均成功率、四集合平均（仅四集合正式评测）、响应时间与显存。可以切换任务集合，或下载完整报告、数据表和原始对比图。

默认选择三模型、原始精度/RTN、Spatial 冒烟档位，共 6 回合，仅验证流程。GPU 默认 0，需根据机器实际空闲情况选择；若忙碌则等待或阻塞，不会终止其他实验。FastWAM 固定 first-frame。AWQ/HSA 显示待接入且禁止提交，没有静默退回其他方法。

支持的是配置中已注册的模型与 checkpoint；页面不支持任意模型上传，也不接受通过 API 注入脚本、命令、解释器或任意配置路径。更换兼容权重由维护者修改启动配置后创建新运行；新架构仍需适配。

## API

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/capabilities` | 模型、方法状态、集合与页面会话 token |
| POST | `/api/preview` | 校验选择，返回阶段/回合规模；不创建任务 |
| GET / POST | `/api/runs` | 列出运行 / 创建并启动一次运行 |
| GET | `/api/runs/{id}` | 当前阶段、完成数、执行状态与原因 |
| GET | `/api/runs/{id}/logs?stage={job_id}` | 指定阶段最近 64 KiB 日志；省略 stage 返回控制器日志 |
| POST | `/api/runs/{id}/cancel` | 写入停止请求，由控制器停止自己的工作进程 |
| POST | `/api/runs/{id}/resume` | 按冻结计划恢复，重新校验输入与已完成结果 |
| GET | `/api/runs/{id}/report` | 通过保存的结果哈希校验后返回报告 JSON |
| GET | `/api/runs/{id}/report/{file}` | 下载 report.md、cells.csv、tradeoffs.png 或 tradeoffs.svg |

创建/预览请求：

```json
{
  "models": ["vla_adapter"],
  "methods": ["rtn"],
  "suites": ["libero_spatial"],
  "profile": "smoke",
  "gpu": 3,
  "wait_seconds": 0
}
```

续跑仅接受 `gpu`、`wait_seconds`；停止请求体为 `{}`。写请求使用 `Content-Type: application/json`，并把 capabilities 返回的 `csrf_token` 放入 `X-Embodiq-Token` 请求头。页面自动处理。服务检查本地 Host、Origin 和 token；这不是多用户认证或公网部署方案。

运行编号仅在 `--runs-root` 下解析。状态来自落盘 `run.json`；不会把网页连接是否在线当作模型任务是否完成。创建/续跑返回 HTTP 202，表示控制器已启动；后续必须查状态，不能将 202 当作评测成功。校验失败为 400，重复运行或报告未就绪为 409。

## 执行与恢复约定

- 网页服务在独立进程中启动控制器；控制器使用对应模型的隔离 Conda 环境。关闭浏览器、停止或重启 HTTP 服务不会自动停止正在执行的评测。
- `.web-dispatch.lock` 由控制器继承，覆盖启动前后窗口；已有 `.controller.lock` 和 GPU 锁仍生效。重启网页服务后，继续指向相同运行目录即可读回记录和锁状态。
- 控制器异常退出且不再持锁时，网页显示“控制器已退出”；不会改写实验原始状态。若遗留工作进程仍持控制器锁，则拒绝再次运行，需要等待退出或由维护者核查。
- 停止请求只触及当前运行；不会终止其他用户的 GPU 进程。启动窗口收到的停止请求也保留，不会被续跑控制器清掉。
- 续跑仍重新校验源码、权重、环境、输入和已完成阶段哈希。代码变更后应创建新运行，不能把旧结果拼入新实验；既有完成报告可按保存的计划与结果哈希只读查看。
- 报告读取验证计划和全部已完成阶段输出，展示保存时结果，不重新运行模型或重算成功率。浏览器不会从其他运行目录补齐空缺。
- 冒烟报告有明确标识，不生成四集合正式宏平均。`fp16` 配置名对应实际 BF16，显存区分稳态 NVML 和加载期 allocator 峰值。
- 当前为单机、每运行顺序执行。不同运行依靠 GPU 锁竞争；等待不是 FIFO 集群队列。不包含 RLark/Kubernetes、多用户权限、任意模型上传或 RoboColiseum 接入。

## 验证

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /data/jyc/tools/anaconda3/envs/openvla/bin/python -m pytest -q tests/test_web.py
```

API 测试使用真实 HTTP 服务与独立控制器进程，GPU/模型操作由 CPU fixture 替换，覆盖提交、停止、续跑、控制器重建、重复启动拦截、非法输入、待接入方法、路径边界、日志截断与报告篡改拒绝。真实模型验收另行记录，不能将 fixture 成功率当成实验数据。

浏览器检查使用本机已有 Chromium / Playwright，覆盖桌面与 390 px 手机宽度、无页面脚本错误、完整正式计划规模预览，以及已有六配置报告展示。没有添加 Playwright 为产品依赖。


本次验收结果：API 专项 **19 通过**；全量回归 **227 通过、23 跳过**。浏览器真实提交
`VLA-Adapter + RTN + Spatial + smoke`，在 GPU 3 完成闭环评测、资源专测、报告三个阶段，
1 回合成功且报告无准入问题。运行编号 `20261009_225331_723f9a2d31`，报告见
[本次冒烟结果](../results/platform/20261009_225331_723f9a2d31/report/report.md)。
本次未重跑正式 12,000 回合；Web/API 覆盖三模型 × 两种可用方法，真实新增运行只验证上述一个组合。
测试日志、桌面/手机截图和机器可读验收记录见
[验收目录](../results/framework_closeout_20261009/web_v1/acceptance.json)。


## 界面第二版：中文实验管理

本轮仅修改静态界面与说明，模型推理、量化配置、结果数据、报告下载文件和后端执行协议保持原有语义。

参考来源（2026-10-09 查阅）：

- [RLark 控制台导航](https://rlark.readthedocs.io/zh-cn/latest/user-guide/console/)与[任务管理](https://rlark.readthedocs.io/zh-cn/latest/user-guide/jobs/)：参考独立导航、任务列表、任务详情、中文状态解释与日志入口。
- [RoboColiseum / Genie Sim 官方说明](https://github.com/AgibotTech/genie_sim#32--benchmark--debug-locally-compete-globally-on-robocoliseum)与[分榜结果接口](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/skills/robocoliseum/challenge-ranking/SKILL.md)：参考按能力维度浏览结果、从汇总定位分项。我们的维度仍是 LIBERO 的四个任务集合，不套用 RoboColiseum 分数。

部分官方截图与线上榜单页面未能加载，设计依据是可读取的官方文档和接口说明；没有复制其品牌、样式代码，也没有新增集群管理或线上榜单后端。

布局调整为“新建评测 / 任务管理 / 评测结果”三个入口。采用灰白底色、紧凑表格和少量蓝色强调；移除口号式标题、英文装饰标题和装饰编号。模型及方法的正式名称保留，程序字段转换为中文：

| 原显示 | 当前显示 |
|---|---|
| smoke / 冒烟验证 | 快速检查 |
| rollout / 程序阶段编号 | 机器人任务评测，并显示模型、方法、任务集合 |
| resource | 速度与显存测量 |
| fp16 | 原始精度，实际 BF16 说明置于展开项 |
| RTN | 四位量化（RTN） |
| GPU | 显卡 |
| NVML 峰值 | 运行峰值显存，说明由显卡监测接口统计 |
| 四集合宏平均 | 四集合平均成功率，并说明正式评测条件 |

运行编号、原始日志与技术限制可展开查看。结果页使用中文对比条形图，按原始结果数值展示；查询时间以毫秒计，显存仍为 GiB（没有将 GiB 改标签冒充 GB）。全部集合页的显存取所选集合峰值中的最大值，响应时间取各集合平均查询耗时的平均值；单集合页直接展示该集合数据。

验收：浏览器检查覆盖三个页面及 390 / 768 / 1024 / 1440 像素宽度，页面无脚本错误或横向溢出；验证空选择拦截、待接入方法不可选、12,000 回合正式计划预览、实验搜索、日志展开、按集合查看结果。浏览器验收仅预览和读取既有结果，没有提交新的 GPU 评测。API 回归 19 项通过。证据见 `results/framework_closeout_20261009/ui_v2/`。
