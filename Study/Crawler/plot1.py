#coding=utf-8
from __future__ import unicode_literals

from pyecharts import Line
from pyecharts.conf import PyEchartsConfig
from pyecharts.engine import EchartsEnvironment
from pyecharts.utils import write_utf8_html_file

attr = ["2018/03/11", "2018/03/18", "2018/03/25", "2018/04/01", "2018/04/08"]
selectedColumn = "Visitor Count"
column = 'Column: {}'.format(selectedColumn)
line = Line(column)
v1 = [661.0, 359.0, 358.0, 536.0, 391.0]
v2 = [102.0, 906.0, 84.0, 878.0, 115.0]
line.add("VP1", attr,v1 ,mark_point=["average", "max", "min"])
line.add("VP2", attr, v2,mark_point=["average", "max", "min"])
line.add("Filter", attr, [430]*len(v1), is_fill=True, 
         area_opacity=0.3, is_smooth=True)#,is_toolbox_show=False
config = PyEchartsConfig(echarts_template_dir='./')
env = EchartsEnvironment(pyecharts_config=config)
tpl = env.get_template('chart_template.html')
html = tpl.render(line=line)
write_utf8_html_file('chart_out.html', html)