from pyecharts import Pie
from __future__ import unicode_literals
from pyecharts import Line
from pyecharts.conf import PyEchartsConfig
from pyecharts.engine import EchartsEnvironment
from pyecharts.utils import write_utf8_html_file
from pyecharts import ThemeRiver


attr=['a','b','c','d']
v1=[40000, 50000, 60000, 70000]
pie =Pie("饼图-星级玫瑰图示例",title_pos='center',width=900)
pie.add("7-17",attr,v1,center=[75,50],is_random=True,radius=[30,75],rosetype='area',is_legend_show=False,is_label_show=True)
#pie.render()
config = PyEchartsConfig(echarts_template_dir='./')
env = EchartsEnvironment(pyecharts_config=config)
#tpl = env.get_template('chart_template.html')
#html = tpl.render(pie=pie)
write_utf8_html_file('chart_out.tif')

##
from pyecharts import Gauge 
gauge =Gauge("仪表盘示例")
gauge.add("业务指标", "完成率", 66.66)
gauge.show_config()
gauge.render()