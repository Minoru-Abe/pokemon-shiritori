# スクレイピング
from bs4 import BeautifulSoup
from urllib import request


LINE_FEED = '\n'

pokemons = []
# Pokemonの現在の最大数を記載。wikipedia上で???になっているものは含めない
poke_num = 905
# スクレイピング
html = request.urlopen("https://ja.wikipedia.org/wiki/%E5%85%A8%E5%9B%BD%E3%83%9D%E3%82%B1%E3%83%A2%E3%83%B3%E5%9B%B3%E9%91%91%E9%A0%86%E3%81%AE%E3%83%9D%E3%82%B1%E3%83%A2%E3%83%B3%E4%B8%80%E8%A6%A7")
soup = BeautifulSoup(html, "html5lib")
tables = soup.find_all("td")

for table in tables:
    pokes = table.find_all("td")
    for i,poke in enumerate(pokes):
        if i%2 == 1:
            #LINE_FEEDが含まれているのでそれを削除
            text_without_linefeed = poke.text.replace(LINE_FEED,'')
            #テーブルの空白要素が入っているのでそれも削除
            if text_without_linefeed != '\xa0':
                pokemons.append(text_without_linefeed)
pokemons = pokemons[:poke_num]


kw = [pokemon for pokemon in pokemons]
# 処理1 記号を日本語に
kw[28] = "ニドランメス"
kw[31] = "ニドランオス"
kw[232] = "ポリゴンツー"
kw[473] = "ポリゴンゼット"

# 処理2　濁音、拗音を処理、"ー"を除去
# 2022.04.03 ァ、ゥ、ェ、ォについても処理を加える
d = {i:j for i, j in zip('ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポァィゥェォャュョ',
                         'カキクケコサシスセソタチツテトハヒフヘホハヒフヘホアイウエオヤユヨ')}
kw = [''.join(d.get(c, c) for c in s.rstrip('ー')) for s in kw]


from pulp import * # pip install pulp
import numpy as np
n, r = len(kw), range(len(kw))
m = LpProblem(sense=LpMaximize) # 数理モデル
x = [[0 if kw[i][-1] != kw[j][0] else LpVariable('x%d_%d'%(i,j), cat=LpBinary)
      for j in r] for i in r] # kw_i から kw_j に繋げるかどうか (1)
y = [LpVariable('y%d'%i, lowBound=0) for i in r] # kw_iが先頭かどうか (2)
z = [LpVariable('z%d'%i, lowBound=0) for i in r] # kw_iの順番 (3)
m += lpSum(x[i][j] for i in r for j in r) # なるべく繋げる (0)
for i in r:
    cou = lpSum(x[i][j] for j in r) # kw_i から出る数
    cin = lpSum(x[j][i] for j in r) # kw_i へ入る数
    m += cou <= 1 # kw_i から出る数は1以下 (4)
    m += cin <= 1 # kw_i へ入る数は1以下 (5)
    m += cou <= cin + y[i] # yに関する制約 (6)
    for j in r:
        m += z[i] <= z[j]-1+(n+1)*(1-x[i][j]) # zに関する制約 (7)
m += lpSum(y) == 1 # 先頭は1つだけ (8)
#%time m.solve() # 求解
print('solving start')
result = m.solve() # 求解
print('solving end')
print('return code = {}'.format(result))
print(int(value(m.objective)) + 1) # 最長しりとり数
rr = range(1,n+1)
vx = np.vectorize(value)(x).astype(int)
i, s = 0, int(np.vectorize(value)(y)@rr)
while s:
    if i:
        print(' -> ', end='')
    i += 1
    print('[%d]%s'%(i,kw[s-1]), end=' ')
    s = vx[s-1]@rr
