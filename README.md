# quant424

An Empirical Study on Fundamental Alphas and Aggregations

基于基本面量化的因子合成实证研究
# 项目结构
```quant424
.
├── images/
│   ├── COMB.jpg 中性化残差组合策略收益
│   ├── ORIG.jpg 原始因子组合策略收益
│   ├── RankIC.jpg 单因子RankIC示意图
│   ├── descri.JPG 描述性统计
│   ├── factor_combo.png 组合后结果
│   ├── factor_explanation.JPG 因子解释
│   └── single_factor.jpg 单因子策略收益
├── report/
│   ├── main.tex 报告源文件
│   ├── ref.bib 参考文献
│   └── 量化投资+第5组+基于基本面量化的因子合成实证研究.pdf
├── src/
│   ├── agg/
│   │   ├── alpha_agg_clean.ipynb 因子组合(6方式)
│   │   └── alpha_agg_nn_clean.ipynb 因子组合(神经网络)
│   ├── backtest/
│   │   ├── backtest[comb1].ipynb 中性化残差组合策略回测
│   │   ├── backtest[orig5].ipynb 原始因子组合策略回测
│   │   ├── backtest[raw].ipynb 原始因子策略回测
│   │   ├── backtest[resid].ipynb 中性化残差策略回测
│   │   └── backtest[xgb].ipynb
│   ├── factors/
│   │   ├── factors估值.py
│   │   ├── factors偿债能力.py
│   │   ├── factors每股指标.py
│   │   ├── factors盈利成长营运效率.py
│   │   └── factors费用率.py
│   └── image_gen/
│       └── 绘图.ipynb
├── LICENSE
├── README.md
├── test.ipynb
└── 量化投资+第5组+基于基本面量化的因子合成实证研究.pdf
```

# 说明
本报告使用github作为仓库地址，详情可见[这里](https://github.com/Silkdust/quant424)。此外，本研究所使用的详细回测结果已上传到阿里云盘，详情可点击[这里](https://www.aliyundrive.com/s/j2GAcXna8id)查看。