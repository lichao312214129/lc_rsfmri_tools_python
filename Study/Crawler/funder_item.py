# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 11:01:36 2018

@author: lenovo
"""

from scrapy.item import Item, Field
 
class FundsortItem(Item):
    # define the fields for your item here like:
    # name = Field()
    pass
 
class FundItem(Item):
    id=Field()
    itemname = Field()
    school = Field()
    subcode=Field()
    fundmoney=Field()
    subcategory=Field()
    time=Field()
    principal=Field()
    url=Field()
    page=Field()