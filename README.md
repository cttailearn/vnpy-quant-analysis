# VeighNa 量化分析系统

基于 VeighNa 量化框架的 A 股技术指标分析与 MA 均线交叉策略回测系统。

## 功能特性

- **技术指标**：MA/EMA 移动平均、MACD、RSI、KDJ、布林带、ADX、ATR、CCI、威廉指标、PSY、OBV 等
- **MA 交叉回测**：支持自定义快线/慢线周期，模拟历史买卖信号
- **回测指标**：年化收益率、夏普比率、最大回撤、胜率、盈亏比、Alpha/Beta 等
- **交易信号**：自动生成多空信号及操作建议

## 技术栈

- **前端**：原生 HTML/CSS/JS，无框架依赖
- **后端**：Python Flask + Gunicorn
- **数据源**：akshare（东方财富）
- **服务器**：Nginx 反向代理

## 快速部署

### 后端安装

```bash
cd stock-api2
python3 -m venv venv
source venv/bin/activate
pip install flask akshare gunicorn

# 启动服务
python main.py
```

### Nginx 配置

```nginx
location /stock-api2/ {
    proxy_pass http://127.0.0.1:18792/;
}
```

## 数据说明

- 回测数据来自 akshare 日 K 线（前复权）
- 佣金费率默认 0.25%（含双向佣金+滑点）
- 初始资金 10 万元

## 免责声明

本系统仅供技术分析参考，不构成任何投资建议。历史回测结果不代表未来收益。
