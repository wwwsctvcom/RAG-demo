# RAG
本项目提供了一种简单易用的向量化文本的技术，同时提供demo供用户使用，RAG技术是解决大模型训练成本高，并且无法获取最新的信息，从而正确的回答用户的问题而提出来的一种技术；

# Vectorize
- 本项目中向量化主要使用chatGPT中的aide-text-embedding-ada-002-v2模型；
> fast_query
```
queried text: 孟加拉国举行2024年孟加拉国大选，因在野党抵制该次选举，谢赫·哈西娜领导的执政党孟加拉国人民联盟大胜。
queried text: 第81届金球奖举行，《奥本海默》获得最佳剧情片在内的五项大奖，成为当晚的大赢家。
fast query time: 0.002000093460083008
```
- 向量化，是将文本等内容处理成一串数字，方便用于计算文本之间的相似度，同时也便于存储，项目提供了两种模式分别为fast_query和query两种；
> query
```
queried text: 第81届金球奖举行，《奥本海默》获得最佳剧情片在内的五项大奖，成为当晚的大赢家。
queried text: 孟加拉国举行2024年孟加拉国大选，因在野党抵制该次选举，谢赫·哈西娜领导的执政党孟加拉国人民联盟大胜。
normal query time: 0.03699994087219238
```
可以看到本项目中提供的两种query模式在速度上差异还是比较明显的，特别是当用户量和用户使用量大的时候，在速度上是会有比较明显的差异的；

# RAG Usage
项目中对使用RAG技术和未使用RAG技术进行了对比，由于LLM对最新的信息没有很好的提供支持，所以需要对大模型提供一个预知的信息，从而更好的回答用户的问题；
> 未使用RAG技术，同时使用chatGPT进行回答，效果如下：
```
问：第81届金球奖举行，获得最佳剧情片在内的五项大奖，成为当晚的大赢家？
答：截至我所知的信息，第81届金球奖（Golden Globe Awards）尚未举行，因此无法提供获得最佳剧情片在内的五项大奖的影片信息。金球奖通常在每年的1月举行，颁奖典礼由好莱坞外国记者协会（Hollywood Foreign Press Association, HFPA）主办，旨在表彰电影和电视业的杰出成就。
请注意，我的信息可能不是最新的，因此建议查看最新的新闻报道或官方公告以获取最新的金球奖信息。
````
> 使用RAG技术，同时使用chatGPT进行回答，效果如下：
```
问：第81届金球奖举行，获得最佳剧情片在内的五项大奖，成为当晚的大赢家？
答：根据您提供的信息，第81届金球奖中，《奥本海默》赢得了最佳剧情片在内的五项大奖，成为当晚的大赢家。这意味着《奥本海默》在该届金球奖中表现非常出色，不仅获得了最佳剧情片这一重要奖项，还赢得了其他四项大奖。不过，您没有提供这四项大奖的具体信息，所以我无法告诉您《奥本海默》还获得了哪些奖项。通常情况下，电影在颁奖典礼上可能会赢得包括最佳导演、最佳编剧、最佳男主角、最佳女主角等在内的奖项。
```
