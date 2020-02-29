# -*- coding: utf-8 -*-
import scrapy
import pandas as pd
import numpy as np
from movie.items import MovieItem

class MeijuSpider(scrapy.Spider):
    name = 'meiju'
    allowed_domains = ['meijutt.com']
    start_urls = ['https://www.jobmd.cn/work/Postdoctoral.htm']

    def parse(self, response):
        allInfo = response.xpath("//dl[@class='rm-dl mb30']")
#        allInfo.get_attribute('href')
        
        name = allInfo.xpath("//li[@class='w-li1']/a[@class='rm-name']/@title").extract()
        Company = allInfo.xpath("//li[@class='w-li1']/a[@class='rm-company']/@title").extract()
        location = allInfo.xpath("//li[@class='w-li2']/span/text()").extract()
        http=allInfo.xpath("//li[@class='w-li4']/span/a/@href").extract()
#        all_info=http
        # to DataFrame
        all_info=np.array([name,Company,location,http]).T
#        all_info=[name,Company,location,http]

#        all_info=pd.DataFrame(all_info,columns=['职位','单位','地点','链接'])
        item = MovieItem()
        item['name'] = Company
#        print(type(item['name']))
        print('=======\n'+item['name'][0][0][0][0])
#        print('========='+item['name'][0]+'==========')
#        item['name'].to_excel('test.xlsx')
#        for info in Company:
#            item = MovieItem()
#            item['name'] = info
#            yield item
        
