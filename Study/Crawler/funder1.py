# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 11:01:09 2018

@author: lenovo
"""

# -*- coding: utf-8 -*-
 
from scrapy.selector import Selector
from fundsort.items import FundItem
from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.http import Request
import re
 
class FundSpider(CrawlSpider):
    name = "fund"
    id = 0
    allowed_domains = ["fund.sciencenet.cn"]
    start_urls =["http://fund.sciencenet.cn/index.php/search/project?name=&person=&no=&company=%E5%8C%97%E4%BA%AC%E5%A4%96%E5%9B%BD%E8%AF%AD%E5%A4%A7%E5%AD%A6&subject=&money1=&money2=&startTime=2005&endTime=2015&subcategory=&redract_url=&submit.x=0&submit.y=0&page=1"
    ]#每换个大学换一次star_urls. url结尾一定是以“page=”结束，
 
    def parse_item(self, response):
        item = response.meta['item']
        sel = Selector(response)
        num=self.getid()
        num=str(num)
        item['id']=num
        item['school'] = sel.xpath('//tbody/tr[2]/td[@colspan="2"]/text()').extract()
        item['subcategory']=sel.xpath('//table[@class="tb no_print"]//tbody//tr[1]/td[@colspan="4"]/text()').extract()
        subcode=sel.xpath('//table[@class="tb no_print"]//tbody//tr[1]/td[@colspan="4"]/text()').extract()[0]
        #subcode=str(subcode)
        item['subcode']=re.findall('([A-M]\d\d)',subcode)
        item['itemname'] = sel.xpath('//div[@class="v_con"]//h1/text()').extract()
        item['fundmoney'] = sel.xpath('//table[@class="tb no_print"]//tbody//tr[3]/td[1]/text()').extract()
        item['time'] = sel.xpath('//table[@class="tb no_print"]//tbody/tr[3]/td[3]/text()').extract()
        item['principal'] = sel.xpath('//table[@class="tb no_print"]//tbody//tr[2]/td[1]/text()').extract()
        item['url']=response.url
        return item
    def getid(self):
        self.id += 1
        return self.id
 
    def parse_link(self,response):
        sel = Selector(response)
        item = response.meta['item']
        items_url=sel.xpath('//div[@id="resultLst"]//div[position()>0]//p//a//@href').extract()
        for item_url in items_url:
            item_url=str(item_url)
            yield Request(url=item_url,meta={'item':item}, callback=self.parse_item)
 
    def parse(self, response):
        urlline=[]
        for i in range(1,17):      #此处换page的。range(1,17)表示1,2···，16  第17页不爬取
          i=str(i)
          a='http://fund.sciencenet.cn/index.php/search/project?name=&person=&no=&company=%E5%8C%97%E4%BA%AC%E5%A4%96%E5%9B%BD%E8%AF%AD%E5%A4%A7%E5%AD%A6&subject=&money1=&money2=&startTime=2005&endTime=2015&subcategory=&redract_url=&submit.x=0&submit.y=0&page='
          #同start_urls
          a=a+i
          urlline.append(a)
 
        for middle_url in urlline:
            item = FundItem()
            middle_url=str(middle_url)
            s=re.findall('page=.{1,3}',middle_url)
            s=str(s)
            item['page']=re.findall('\d{1,3}',s)
            yield Request(middle_url,meta={'item':item},callback=self.parse_link)