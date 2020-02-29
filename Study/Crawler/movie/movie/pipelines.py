# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

class MoviePipeline(object):
    def process_item(self, item, spider):
        with open("mymeiju.txt",'a') as fp:
            fp.write(item['name'][0][0][0][0]+'\n')
#        print(item['name'])
#        all_info=pd.DataFrame(item['name'],columns=['单位'])
#        item['name'].to_csv('test.txt')
