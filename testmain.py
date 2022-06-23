import nee, stockRank


#주식 거래량 상위 20위 
stocksList = stockRank.getStocks()

#뉴스 헤드라인 크롤링 > 엑셀파일로 저장 
nee.start(stocksList)