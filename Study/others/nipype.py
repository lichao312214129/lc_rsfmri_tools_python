# -*- coding: utf-8 -*-
import nipype.interfaces.io as nio
# 生成一个DataGrabber对象
ds = nio.DataGrabber()
# 指定根目录
ds.inputs.base_directory = r'D:\myCodes\MVPA_LC\Python\study\workstation'
# 指定模板，用来过滤文档
ds.inputs.template = 's*/ses-test/func/sub-01_ses-retest_task-covertverb.gz'
# 需要注意的是，此处并没有对文档进行排序，如果需要排序需要指定sorted参数
ds.inputs.sort_filelist = True
# 执行
results = ds.run()
# 查看执行结果
results.outputs
#
# 在生成DataGrabber对象的时候，需要指定infields参数
ds = nio.DataGrabber(infields=['subject_id'])
ds.inputs.base_directory = '/data/ds000114'
# 使用%02d来设计模板
ds.inputs.template = 'sub-%02d/ses-test/func/*fingerfootlips*.nii.gz'
ds.inputs.sort_filelist = True
# 指定subject_id的具体数值
ds.inputs.subject_id = [1, 7]
results = ds.run()
results.outputs


# another method

from nipype import SelectFiles, Node

templates = {'func': 'sub{subject_id1}\\ses-{ses_name}\\func\\sub-{subject_id2}_ses-{ses_name}_task-{task_name}.gz'}

# Create SelectFiles node
sf = Node(SelectFiles(templates),
          name='selectfiles')

# Location of the dataset folder
sf.inputs.base_directory = r'D:\myCodes\MVPA_LC\Python\study\workstation'

# Feed {}-based placeholder strings with values
sf.inputs.subject_id1 = '00[1,2]'
sf.inputs.subject_id2 = '01'
sf.inputs.ses_name = "retest"
sf.inputs.task_name = 'covertverb'
path=sf.run().outputs.__dict__


