# Mikomiko Tagger 评测指标手册

> 评测对象:Qwen3.5-2B 全参数 SFT 图像 tagger(mikomiko 数据)。
> 评测集:400 张 = unseen 200 + stratified 200。**分组真实含义(2026-07-09 按图片/post 重叠核实)**:
> unseen 与训练集 post 级零重叠 = 真泛化;stratified 图片级与训练集零重叠、但 post 级重叠
> 1812/1814(同图集/场景的另一张)= "近似见过"+稀有 tag 分层,既非纯泛化也非纯记忆。
> 纯记忆探针另见 `train_seen_mini`(训练集原样随机 200 条,seed=42,跑 `test_mikomiko_seen.sh`)。
> **gold = 标准答案**,即评测图在平台上已有的 tag 列表。注意 gold 本身非穷尽、不一致
> (同一概念有图标了有图没标),所以全部分数都是**下界**。
> 结果文件:`saves/qwen3.5-2b/mikomiko/predict_sanity/runs/evalmini_step_*/metrics.json`,
> 历史曲线 `predict_sanity/evalmini_history.tsv`。

## 指标定义(一张大表)

基础记号:对每张图,**TP** = 预测且在 gold 里的 tag 数,**FP** = 预测了但 gold 没有,**FN** = gold 有但漏了。
tag 分两种形态:**atom** = 单概念 tag(`Blonde`、`MILF`);**composite** = 多词组合 tag(`Blonde MILF`、`Big Ass Latina`)。

| 指标 | 一句话含义 | 计算口径 | 松紧度 | 怎么读 |
|---|---|---|---|---|
| microP | 预测的 tag 里有多少是对的 | 400 图所有 tag 汇总后 TP/(TP+FP) | 严(整 tag 精确匹配) | 低 = 话多说错/多说 |
| microR | gold 的 tag 有多少被找回 | 汇总后 TP/(TP+FN) | 严 | 低 = 该说的漏了 |
| **microF1** | **主指标**:P 和 R 的调和平均 | 2PR/(P+R),tag 池全局统计,tag 多的图权重大 | 严 | 综合分;P/R 偏科都上不去 |
| macroF1 | 每图各算 F1 再对 400 图平均 | 图级等权平均 | 严 | 与 microF1 接近 = 没有少数图特别拖后腿 |
| atomF1 | 只看单概念 tag 的 F1 | 仅 atom 参与匹配 | 严 | 衡量"看图识概念"的基本功 |
| compF1_exact | 复合 tag 一字不差才算对 | 仅 composite,整串精确匹配 | 最严 | 低 = 组合形态背不对 |
| compF1_subset | 复合 tag 的每个词在对面出现即算对 | 仅 composite,词集合覆盖即记对 | 宽 | 与 exact 的差值 = "概念对、形态错"的量 |
| tokF1 | 把所有 tag 打散成单词再对 | 词级 F1,不管组合 | 最宽 | 概念层面的能力上限估计 |
| pred_tpi / gold_tpi | 每图平均预测 tag 数 vs gold tag 数 | 计数 | — | pred≫gold = 过度生成(over-tagging) |
| pred_cpi / gold_cpi | 每图平均**复合** tag 数 vs gold | 计数 | — | 复合 tag 的过度生成检查 |

> BLEU-4 / ROUGE 已从报告中剔除:它们是序列指标,对无序 tag 集合无意义。

## 评测结果(step 11530 = 1.33 epoch vs step 17296 = 2.0 epoch,cosine 跑满)

| 分组 | step | microP | microR | microF1 | macroF1 | atomF1 | compEx | compSub | tokF1 | pred/gold tpi | pred/gold cpi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL (400) | 11530 | 36.2 | 39.7 | 37.9 | 37.1 | 49.6 | 22.3 | 35.0 | 48.0 | 11.9 / 10.9 | 5.7 / 4.1 |
| ALL (400) | 17296 | 36.9 | 40.1 | **38.4** | 37.6 | 49.9 | 23.1 | 35.7 | 48.2 | 11.8 / 10.9 | 5.6 / 4.1 |
| unseen (200) | 11530 | 37.8 | 40.8 | 39.2 | 38.3 | 50.9 | 22.6 | 36.0 | 49.0 | 12.0 / 11.1 | 5.6 / 4.0 |
| unseen (200) | 17296 | 38.7 | 41.5 | 40.0 | 39.3 | 51.8 | 23.3 | 36.5 | 49.5 | 12.0 / 11.1 | 5.6 / 4.0 |
| stratified (200) | 11530 | 34.6 | 38.6 | 36.5 | 35.8 | 48.1 | 22.1 | 34.1 | 46.9 | 11.8 / 10.6 | 5.8 / 4.2 |
| stratified (200) | 17296 | 35.0 | 38.7 | 36.8 | 35.9 | 47.8 | 22.9 | 34.8 | 46.9 | 11.7 / 10.6 | 5.7 / 4.2 |
| **train_seen (200)** | 17296 | 35.9 | 38.4 | 37.1 | 36.0 | 47.7 | 23.7 | 35.6 | 48.4 | 11.6 / 10.8 | 5.6 / 4.3 |

## 当前读数的三个结论

1. **tokF1≈48 而 microF1≈38**:模型识概念的能力有五成,中间 10 个点耗在复合 tag 形态
   对不上(compEx 23 vs compSub 36)和过度生成(复合 tag 多说 38%:5.6 vs 4.1/图)。
   丢分大头是标签格式问题,不是视觉能力问题。
2. **加训收益在噪声内**:1.33→2.0 epoch(+50% 步数)microF1 仅 +0.5pt,400 样本抽样噪声
   约 ±1~1.5pt;stratified(稀有 tag 难尾)基本不动,atomF1 甚至微降。
3. **FP/FN 头部是同一批 tag**(brunette、natural tits、amateur 两边都在):gold 标注
   不一致的铁证,继续在脏 gold 上堆步数是在学标注员的遗漏习惯。
4. **记忆探针结果(2026-07-09,seen_step_17296)**:train_seen 37.1 ≈ stratified 36.8 ≈
   unseen 40.0,三层探针无梯度 —— 模型对原样训过 2 epoch 的图没有任何记忆优势。
   残差是标注噪声(aleatoric)而非欠拟合(epistemic),**"继续训练提分"这条路正式关闭**;
   ~50 的 atomF1 就是这套标签的一致性上限。

## 决策建议(2026-07-09)

- 先对全量测试集(unseen 122,870 + stratified 1,945)跑 predict 补统计功效;
- 人工抽审 50–100 个 FP,量化"gold 缺标"占比;
- 标签空间治理(alias 归并、主观 tag 剔除、composite 白名单或只训 atom + 规则拼装)
  优先于继续训练;若重训,重启新 cosine(勿在已衰减的 run 上 resume)。
