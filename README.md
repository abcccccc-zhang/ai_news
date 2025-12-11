# ai_news
ai 分析 新闻

通过dockerrun 里面的命令运行，compose文件自行在里面加变量

本项目基于github的TrendRadar 做了修改。添加了如下功能

1.推送新闻前通过ai 判断是否属于我需要的信息

2.分析前判断内容是否和我需要的信息有关，双重判断，因为用免费的api，所以没关系

3.最后使用了langgraph的node通过另一个模型去分析内容，这个每天有免费额度。
