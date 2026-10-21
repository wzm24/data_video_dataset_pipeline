340 条 OK 五件套的合并副本，每条一个文件夹，名字自带类别前缀（bar_ / line_ / area_ / donut_ / pie_），按名字排序即按类别分组。

每条目录里六个文件：
  keyframe.png        原视频关键帧
  semantic.svg        我们确认过的语义图（风格的唯一真值）
  data_table.csv      数据表
  narration.jsonl     原视频旁白
  intent.json         意图（只取 intent_en 作为公开输入）
  g1_draft_b2.json    G1 必需语义单元草稿

原始位置仍是 data/generated/{bar,line,area,donut,pie}_ok_5files/。
这一份是方便浏览和交付的副本；两份内容一致，改动请只改一边并说明。
