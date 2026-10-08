# 328条诊断集

七种情绪各41题，共287题；抑郁分类任务41题；总计328题。

原320题名单和顺序全部保留，新增8题按固定种子32846和样本身份哈希选取，不参照模型正确性。
原320题采用已知模型结果筛选，因此整体仍是有选择偏差的诊断集，不是独立最终测试集。
更改题数不能消除选择偏差；报告保留筛选来源与全部三组结果。

新增原754条索引：[654, 718, 1, 620, 212, 507, 102, 38]。
标签：{'anger': 41, 'disgust': 41, 'fear': 41, 'happy': 41, 'neutral': 41, 'sad': 41, 'surprise': 41, 'depression_task/depression': 11, 'depression_task/no depression': 30}。
模态：{'text_video_audio': 184, 'text_audio': 103, 'text': 41}。
来源：{'new538': 218, 'legacy216': 110}。
父名单SHA256：4be0a64627db05aeac5f82354c603a02019289a6f62586bf5a1f3b17c2818373。
新名单SHA256：9ab169af8899c7ab35ac5910cb494d2a6e7f479ea8fcb976e86d110af8deac32。
