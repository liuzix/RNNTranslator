r"""
The rules I can state, as a transducer: English -> frame -> Japanese.

Everything is finite and table-driven so it compiles to the same two-layer ReLU cell
as before (compile_rules.py): the parser is a DFA over (phase, token class, a few
register predicates); the frame is a fixed set of registers; generation is a fixed
template whose every slot is a table lookup over register contents and feature bits.

What is NOT here is the point of the experiment: whatever the held-out Tanaka
sentences need that I could not write down as a rule.
"""
import re
from collections import Counter

# ============================================================================
# 1. Japanese morphology I know
# ============================================================================
GODAN_ROWS = {  # dictionary ending -> (a, i, e, o) rows and te/ta ending
    "う": ("わ", "い", "え", "お", "って"), "く": ("か", "き", "け", "こ", "いて"),
    "ぐ": ("が", "ぎ", "げ", "ご", "いで"), "す": ("さ", "し", "せ", "そ", "して"),
    "つ": ("た", "ち", "て", "と", "って"), "ぬ": ("な", "に", "ね", "の", "んで"),
    "ぶ": ("ば", "び", "べ", "ぼ", "んで"), "む": ("ま", "み", "め", "も", "んで"),
    "る": ("ら", "り", "れ", "ろ", "って"),
}

class Pred:
    """A conjugable predicate: kind in v1 (ichidan), v5 (godan), suru, kuru, ia (i-adjective),
    na (na-adjective / noun + copula), fixed (already finite)."""
    def __init__(self, kind, stem, dic=None):
        self.kind, self.stem = kind, stem
        self.dic = dic or (stem + "る" if kind == "v1" else stem)

    # --- stems ---
    def nai_stem(self):
        k = self.kind
        if k == "v1": return self.stem
        if k == "v5": return self.stem[:-1] + GODAN_ROWS[self.stem[-1]][0]
        if k == "suru": return self.stem[:-2] + "し"
        if k == "kuru": return self.stem[:-1]          # 来
        raise ValueError
    def masu_stem(self):
        k = self.kind
        if k == "v1": return self.stem
        if k == "v5": return self.stem[:-1] + GODAN_ROWS[self.stem[-1]][1]
        if k == "suru": return self.stem[:-2] + "し"
        if k == "kuru": return self.stem[:-1]
        raise ValueError
    def te(self):
        k = self.kind
        if k == "forms": return self.stem[0][0][:-1] + "って"
        if k == "v1": return self.stem + "て"
        if k == "v5":
            if self.stem.endswith("行く"): return self.stem[:-1] + "って"
            return self.stem[:-1] + GODAN_ROWS[self.stem[-1]][4]
        if k == "suru": return self.stem[:-2] + "して"
        if k == "kuru": return self.stem[:-1] + "て"
        if k == "ia": return self.stem + "くて"
        if k == "na": return self.stem + "で"
        raise ValueError
    def ta(self):
        t = self.te()
        return t[:-1] + ("だ" if t.endswith("で") else "た") if self.kind in ("v1", "v5", "suru", "kuru") else None

    # --- transforms that yield a new predicate ---
    def potential(self):
        k = self.kind
        if k == "v1": return Pred("v1", self.stem + "られ")
        if k == "v5": return Pred("v1", self.stem[:-1] + GODAN_ROWS[self.stem[-1]][2])
        if k == "suru": return Pred("v1", self.stem[:-2] + "でき")
        if k == "kuru": return Pred("v1", self.stem[:-1] + "られ")
        raise ValueError
    def tai(self):          return Pred("ia", self.masu_stem() + "た")
    def causative(self):
        caus = {"v1": self.stem + "させ", "v5": self.nai_stem() + "せ", "suru": self.stem[:-2] + "させ", "kuru": self.stem[:-1] + "させ"}.get(self.kind)
        return Pred("v1", caus, dic=caus + "る") if caus else self
    def progressive(self):  return Pred("v1", self.te() + "い")
    def should(self):       return Pred("na", self.dic + "べき")
    def maybe(self):        return Pred("fixed", self.dic + "かもしれない")
    def intend(self):       return Pred("na", self.dic + "つもり")
    def volitional(self, polite):
        k = self.kind
        if polite: return self.masu_stem() + "ましょう"
        if k == "v1": return self.stem + "よう"
        if k == "v5": return self.stem[:-1] + GODAN_ROWS[self.stem[-1]][3] + "う"
        if k == "suru": return self.stem[:-2] + "しよう"
        if k == "kuru": return self.stem[:-1] + "よう"
        raise ValueError

    # --- finite forms ---
    def finite(self, past=False, neg=False, polite=False):
        k = self.kind
        if k in ("v1", "v5", "suru", "kuru"):
            if polite:
                s = self.masu_stem()
                return s + ["ます", "ました", "ません", "ませんでした"][2 * neg + past]
            if not neg:
                return self.ta() if past else self.dic
            n = self.nai_stem() + ("なかった" if past else "ない")
            return n
        if k == "ia":
            base = [self.stem + "い", self.stem + "かった", self.stem + "くない", self.stem + "くなかった"][2 * neg + past]
            return base + ("です" if polite else "")
        if k == "na":
            if polite:
                return self.stem + ["です", "でした", "ではありません", "ではありませんでした"][2 * neg + past]
            return self.stem + ["だ", "だった", "ではない", "ではなかった"][2 * neg + past]
        if k == "forms":
            plain, pol = self.stem
            return (pol if polite else plain)[2 * neg + past]
        if k == "fixed":
            return self.stem + ("です" if polite else "")
        raise ValueError(k)

def make_pred(ja, cls):
    if cls == "aru": return Pred("forms", (["ある", "あった", "ない", "なかった"], ["あります", "ありました", "ありません", "ありませんでした"]), dic="ある")
    if cls == "g":  return Pred("v5", ja)
    if cls == "i":  return Pred("v1", ja[:-1], dic=ja)
    if cls == "s":  return Pred("suru", ja)
    if cls == "k":  return Pred("kuru", ja)
    raise ValueError(cls)

# ============================================================================
# 2. Lexicon: the words I can state.  pos: v (verb), n (noun), a (i-adj), na (na-adj),
#    adv, t (time adverb, goes first), p (preposition), int (ignored interjection)
#    verbs: lemma|v|ja|conj|past|participle ; nouns: word|n|ja|class(h human,a animal,t thing,p place)
# ============================================================================
LEXICON_TEXT = """
be|v|いる|i|was|been
each other|n|お互い|h
one another|n|お互い|h
one after another|adv|次々に
again and again|adv|何度も
more and more|adv|ますます
as usual|adv|いつものように
twice a week|adv|週に二回
once a week|adv|週に一回
twice a day|adv|一日に二回
once a day|adv|一日に一回
three times a week|adv|週に三回
every day|adv|毎日
any more|adv|もう
ever since|adv|それ以来
just now|adv|たった今
any longer|adv|もう
next|a|次の
than usual|adv|いつもより
than before|adv|前より
than ever|adv|これまで以上に
long|adv|長く
bus stop|n|バス停|t
stop|n|停留所|t
movie theater|n|映画館|t
small change|n|小銭|t
way of thinking|n|考え方|t
fare|n|運賃|t
a little|det|少しの
a little|adv|少し
as soon as|conj|
so that|conj|
as if|conj|
as though|conj|
so long as|conj|
as long as|conj|
even if|conj|
even though|conj|
in case|conj|
whenever|conj|
every time|conj|
now that|conj|
by the time|conj|
and then|conj|
have|v|持つ|g|had|had
do|v|する|s|did|done
go|v|行く|g|went|gone
come|v|来る|k|came|come
see|v|見る|i|saw|seen
get|v|得る|i|got|got
make|v|作る|g|made|made
know|v|知る|g|knew|known
take|v|取る|g|took|taken
give|v|あげる|i|gave|given
think|v|思う|g|thought|thought
say|v|言う|g|said|said
tell|v|話す|g|told|told
want|v|欲しがる|g
like|v|好む|g
help|v|助ける|i
work|v|働く|g
play|v|遊ぶ|g
read|v|読む|g|read|read
write|v|書く|g|wrote|written
eat|v|食べる|i|ate|eaten
drink|v|飲む|g|drank|drunk
buy|v|買う|g|bought|bought
sell|v|売る|g|sold|sold
pay|v|払う|g|paid|paid
find|v|見つける|i|found|found
lose|v|失う|g|lost|lost
leave|v|出る|i|left|left
arrive|v|着く|g
stay|v|泊まる|g
live|v|住む|g
die|v|死ぬ|g
run|v|走る|g|ran|run
walk|v|歩く|g
swim|v|泳ぐ|g|swam|swum
sleep|v|眠る|g|slept|slept
wake|v|起きる|i|woke|woken
wait|v|待つ|g
speak|v|話す|g|spoke|spoken
talk|v|話す|g
hear|v|聞く|g|heard|heard
listen|v|聞く|g
look|v|見る|i
watch|v|見る|i
call|v|呼ぶ|g
ask|v|尋ねる|i
answer|v|答える|i
meet|v|会う|g|met|met
visit|v|訪ねる|i
show|v|見せる|i|showed|shown
bring|v|持ってくる|k|brought|brought
send|v|送る|g|sent|sent
put|v|置く|g|put|put
keep|v|保つ|g|kept|kept
hold|v|持つ|g|held|held
open|v|開ける|i
close|v|閉める|i
start|v|始める|i
begin|v|始める|i|began|begun
stop|v|止める|i
finish|v|終える|i
end|v|終わる|g
try|v|試す|g
use|v|使う|g
need|v|必要とする|s
learn|v|学ぶ|g
study|v|勉強する|s
teach|v|教える|i|taught|taught
understand|v|理解する|s|understood|understood
remember|v|覚える|i
forget|v|忘れる|i|forgot|forgotten
believe|v|信じる|i
feel|v|感じる|i|felt|felt
hope|v|望む|g
love|v|愛する|s
enjoy|v|楽しむ|g
fail|v|失敗する|s
succeed|v|成功する|s
pass|v|合格する|s
win|v|勝つ|g|won|won
catch|v|捕まえる|i|caught|caught
cut|v|切る|g|cut|cut
break|v|壊す|g|broke|broken
build|v|建てる|i|built|built
grow|v|育つ|g|grew|grown
change|v|変える|i
move|v|動く|g
turn|v|回る|g
sit|v|座る|g|sat|sat
stand|v|立つ|g|stood|stood
fall|v|落ちる|i|fell|fallen
drive|v|運転する|s|drove|driven
ride|v|乗る|g|rode|ridden
fly|v|飛ぶ|g|flew|flown
sing|v|歌う|g|sang|sung
dance|v|踊る|g
cook|v|料理する|s
clean|v|掃除する|s
wash|v|洗う|g
wear|v|着る|i|wore|worn
smoke|v|吸う|g
laugh|v|笑う|g
cry|v|泣く|g
smile|v|微笑む|g
rain|v|雨が降る|g
snow|v|雪が降る|g
happen|v|起こる|g
return|v|帰る|g
enter|v|入る|g
agree|v|賛成する|s
decide|v|決める|i
promise|v|約束する|s
apologize|v|謝る|g
worry|v|心配する|s
mean|v|意味する|s|meant|meant
cost|v|かかる|g|cost|cost
spend|v|使う|g|spent|spent
save|v|貯める|i
borrow|v|借りる|i
lend|v|貸す|g|lent|lent
carry|v|運ぶ|g
climb|v|登る|g
cross|v|渡る|g
miss|v|逃す|g
marry|v|結婚する|s
invite|v|招く|g
join|v|参加する|s
attend|v|出席する|s
choose|v|選ぶ|g|chose|chosen
seem|v|見える|i
sound|v|聞こえる|i
smell|v|におう|g
taste|v|味がする|s
touch|v|触る|g
hit|v|打つ|g|hit|hit
kill|v|殺す|g
hurt|v|傷つける|i|hurt|hurt
fight|v|戦う|g|fought|fought
steal|v|盗む|g|stole|stolen
hide|v|隠す|g|hid|hidden
draw|v|描く|g|drew|drawn
paint|v|塗る|g
check|v|調べる|i
explain|v|説明する|s
describe|v|述べる|i
solve|v|解く|g
discuss|v|話し合う|g
suggest|v|提案する|s
advise|v|忠告する|s
allow|v|許す|g
expect|v|期待する|s
intend|v|つもりだ|fixed
practice|v|練習する|s
travel|v|旅行する|s
continue|v|続ける|i
follow|v|ついて行く|g
reach|v|着く|g
throw|v|投げる|i|threw|thrown
wish|v|願う|g
thank|v|感謝する|s
blame|v|責める|i
remain|v|残る|g
appear|v|現れる|i
grow|v|育つ|g|grew|grown
shine|v|輝く|g|shone|shone
hurry|v|急ぐ|g
apple|n|りんご|t
book|n|本|t
books|n|本|t
car|n|車|t
cars|n|車|t
house|n|家|p
home|n|家|p
school|n|学校|p
station|n|駅|p
money|n|お金|t
time|n|時間|t
day|n|日|t
days|n|日|t
night|n|夜|t
morning|n|朝|t
afternoon|n|午後|t
evening|n|夕方|t
week|n|週|t
month|n|月|t
year|n|年|t
years|n|年|t
hour|n|時間|t
hours|n|時間|t
minutes|n|分|t
father|n|父|h
mother|n|母|h
brother|n|兄|h
sister|n|姉|h
son|n|息子|h
daughter|n|娘|h
uncle|n|おじ|h
aunt|n|おば|h
parents|n|両親|h
wife|n|妻|h
husband|n|夫|h
child|n|子供|h
children|n|子供たち|h
baby|n|赤ちゃん|h
boy|n|少年|h
boys|n|少年たち|h
girl|n|少女|h
man|n|男|h
men|n|男たち|h
woman|n|女|h
people|n|人々|h
person|n|人|h
friend|n|友達|h
friends|n|友達|h
teacher|n|先生|h
student|n|学生|h
students|n|学生たち|h
doctor|n|医者|h
police|n|警察|h
family|n|家族|h
nobody|n|誰も|h
dog|n|犬|a
cat|n|猫|a
bird|n|鳥|a
horse|n|馬|a
fish|n|魚|a
animals|n|動物|a
work|n|仕事|t
job|n|仕事|t
business|n|仕事|t
news|n|ニュース|t
story|n|話|t
letter|n|手紙|t
word|n|言葉|t
words|n|言葉|t
name|n|名前|t
question|n|質問|t
answer|n|答え|t
problem|n|問題|t
idea|n|考え|t
plan|n|計画|t
fact|n|事実|t
truth|n|真実|t
opinion|n|意見|t
advice|n|忠告|t
mind|n|心|t
life|n|人生|t
health|n|健康|t
accident|n|事故|t
trouble|n|困ったこと|t
trouble|v|悩ます|g
worry|n|心配|t
tickets|n|切符|t
ticket|n|切符|t
chance|n|機会|t
stars|n|星|t
matter|n|問題|t
success|n|成功|t
failure|n|失敗|t
mistake|n|間違い|t
mistakes|n|間違い|t
chance|n|機会|t
reason|n|理由|t
way|n|道|t
thing|n|こと|t
things|n|こと|t
nothing|n|何も|t
everything|n|全て|t
something|n|何か|t
anything|n|何か|t
music|n|音楽|t
song|n|歌|t
game|n|試合|t
tennis|n|テニス|t
baseball|n|野球|t
soccer|n|サッカー|t
piano|n|ピアノ|t
movie|n|映画|t
movies|n|映画|t
tv|n|テレビ|t
radio|n|ラジオ|t
picture|n|写真|t
camera|n|カメラ|t
computer|n|コンピューター|t
phone|n|電話|t
telephone|n|電話|t
english|n|英語|t
japanese|n|日本語|t
french|n|フランス語|t
japan|n|日本|p
tokyo|n|東京|p
london|n|ロンドン|p
paris|n|パリ|p
america|n|アメリカ|p
kyoto|n|京都|p
osaka|n|大阪|p
york|n|ニューヨーク|p
city|n|都市|p
town|n|町|p
village|n|村|p
country|n|国|p
world|n|世界|p
room|n|部屋|p
office|n|事務所|p
hospital|n|病院|p
hotel|n|ホテル|p
park|n|公園|p
garden|n|庭|p
library|n|図書館|p
store|n|店|p
shop|n|店|p
bank|n|銀行|p
airport|n|空港|p
street|n|通り|p
road|n|道路|p
bridge|n|橋|p
mountain|n|山|p
river|n|川|p
lake|n|湖|p
sea|n|海|p
beach|n|浜|p
sky|n|空|t
sun|n|太陽|t
moon|n|月|t
rain|n|雨|t
snow|n|雪|t
weather|n|天気|t
spring|n|春|t
summer|n|夏|t
winter|n|冬|t
autumn|n|秋|t
tree|n|木|t
trees|n|木|t
flower|n|花|t
flowers|n|花|t
water|n|水|t
tea|n|お茶|t
coffee|n|コーヒー|t
milk|n|牛乳|t
wine|n|ワイン|t
beer|n|ビール|t
food|n|食べ物|t
bread|n|パン|t
cake|n|ケーキ|t
lunch|n|昼食|t
dinner|n|夕食|t
breakfast|n|朝食|t
dictionary|n|辞書|t
paper|n|紙|t
newspaper|n|新聞|t
novel|n|小説|t
novels|n|小説|t
pen|n|ペン|t
bag|n|かばん|t
box|n|箱|t
hat|n|帽子|t
coat|n|コート|t
dress|n|ドレス|t
shoes|n|靴|t
clothes|n|服|t
door|n|ドア|t
window|n|窓|t
table|n|テーブル|t
chair|n|椅子|t
bed|n|ベッド|t
wall|n|壁|t
floor|n|床|t
train|n|電車|t
bus|n|バス|t
taxi|n|タクシー|t
plane|n|飛行機|t
airplane|n|飛行機|t
bicycle|n|自転車|t
ticket|n|切符|t
fire|n|火事|t
war|n|戦争|t
peace|n|平和|t
history|n|歴史|t
science|n|科学|t
class|n|授業|t
lesson|n|授業|t
exam|n|試験|t
examination|n|試験|t
homework|n|宿題|t
test|n|テスト|t
party|n|パーティー|t
meeting|n|会議|t
trip|n|旅行|t
concert|n|コンサート|t
company|n|会社|t
government|n|政府|t
information|n|情報|t
experience|n|経験|t
knowledge|n|知識|t
effort|n|努力|t
efforts|n|努力|t
heart|n|心|t
head|n|頭|t
hand|n|手|t
hands|n|手|t
eye|n|目|t
eyes|n|目|t
face|n|顔|t
hair|n|髪|t
body|n|体|t
voice|n|声|t
dream|n|夢|t
fever|n|熱|t
cold|n|風邪|t
illness|n|病気|t
medicine|n|薬|t
noise|n|騒音|t
sound|n|音|t
light|n|光|t
age|n|年齢|t
number|n|数|t
price|n|値段|t
cost|n|費用|t
dollars|n|ドル|t
yen|n|円|t
earth|n|地球|t
place|n|場所|p
front|n|前|t
end|n|終わり|t
line|n|線|t
point|n|点|t
sense|n|感覚|t
order|n|順序|t
rule|n|規則|t
case|n|場合|t
future|n|未来|t
past|n|過去|t
moment|n|瞬間|t
smoking|n|喫煙|t
swimming|n|水泳|t
skiing|n|スキー|t
cooking|n|料理|t
reading|n|読書|t
fishing|n|釣り|t
traffic|n|交通|t
danger|n|危険|t
speech|n|スピーチ|t
promise|n|約束|t
beauty|n|美しさ|t
kindness|n|親切|t
glass|n|グラス|t
cup|n|カップ|t
seat|n|席|t
bath|n|風呂|t
look|n|一見|t
care|n|注意|t
part|n|部分|t
headache|n|頭痛|t
fun|n|楽しみ|t
decision|n|決定|t
progress|n|進歩|t
call|n|電話|t
favor|n|お願い|t
dishes|n|皿|t
cards|n|トランプ|t
chess|n|チェス|t
lie|n|嘘|t
diary|n|日記|t
secret|n|秘密|t
record|n|記録|t
weight|n|体重|t
temper|n|機嫌|t
race|n|競争|t
prize|n|賞|t
attention|n|注意|t
visit|n|訪問|t
alarm|n|目覚まし|t
risk|n|危険|t
hello|n|挨拶|t
prayer|n|祈り|t
shopping|n|買い物|t
trumpet|n|トランペット|t
flute|n|フルート|t
drums|n|太鼓|t
good|a|良い
bad|a|悪い
big|a|大きい
large|a|大きい
small|a|小さい
little|a|小さい
old|a|古い
new|a|新しい
young|a|若い
long|a|長い
short|a|短い
high|a|高い
low|a|低い
tall|a|高い
hot|a|暑い
cold|a|寒い
warm|a|暖かい
cool|a|涼しい
fast|a|速い
slow|a|遅い
early|a|早い
late|a|遅い
hard|a|難しい
difficult|a|難しい
easy|a|易しい
cheap|a|安い
expensive|a|高い
busy|a|忙しい
happy|a|嬉しい
sad|a|悲しい
glad|a|嬉しい
angry|fx|怒っている
afraid|a|怖い
tired|fx|疲れている
sick|fx|具合が悪い
ill|na|病気
hungry|fx|お腹が空いている
beautiful|a|美しい
pretty|a|きれいな
interesting|a|面白い
boring|a|退屈な
strong|a|強い
weak|a|弱い
heavy|a|重い
light|a|軽い
dark|a|暗い
bright|a|明るい
white|a|白い
black|a|黒い
red|a|赤い
blue|a|青い
right|a|正しい
wrong|a|間違った
true|a|本当の
kind|a|優しい
nice|a|素敵な
wonderful|a|素晴らしい
great|a|素晴らしい
careful|a|注意深い
clever|a|賢い
foolish|a|愚かな
rich|a|金持ちの
poor|a|貧しい
near|a|近い
far|a|遠い
wide|a|広い
narrow|a|狭い
deep|a|深い
sweet|a|甘い
delicious|a|美味しい
clear|a|明らかな
quiet|na|静か
famous|na|有名
important|na|大切
necessary|na|必要
simple|na|簡単
free|na|自由
safe|na|安全
dangerous|na|危険
strange|na|奇妙
lucky|na|幸運
healthy|na|健康
convenient|na|便利
polite|na|丁寧
energetic|na|元気
fine|na|元気
sure|na|確か
absent|na|欠席
able|na|可能
popular|na|人気
natural|na|自然
different|na|違う
same|na|同じ
ready|fx|用意ができている
very|adv|とても
so|adv|とても
too|adv|も
really|adv|本当に
always|adv|いつも
usually|adv|たいてい
often|adv|よく
sometimes|adv|時々
never|adv|決して
again|adv|また
still|adv|まだ
already|adv|もう
just|adv|ちょうど
only|adv|だけ
also|adv|も
well|adv|上手に
hard|adv|一生懸命
fast|adv|速く
slowly|adv|ゆっくり
early|adv|早く
late|adv|遅く
soon|adv|すぐに
quickly|adv|速く
suddenly|adv|突然
easily|adv|簡単に
carefully|adv|注意深く
together|adv|一緒に
alone|adv|一人で
abroad|adv|海外に
here|adv|ここに
there|adv|そこに
away|adv|離れて
back|adv|戻って
up|adv|上に
down|adv|下に
out|adv|外に
home|adv|家に
almost|adv|ほとんど
quite|adv|かなり
enough|adv|十分に
much|adv|たくさん
more|adv|もっと
far|adv|遠く
yet|adv|まだ
ever|adv|今までに
even|adv|さえ
probably|adv|たぶん
maybe|adv|たぶん
perhaps|adv|たぶん
anymore|adv|もう
today|t|今日
yesterday|t|昨日
tomorrow|t|明日
now|t|今
tonight|t|今夜
then|t|その時
once|t|一度
ago|t|前
everyday|t|毎日
last night|t|昨夜
last week|t|先週
last year|t|去年
next week|t|来週
next year|t|来年
this morning|t|今朝
this evening|t|今晩
this year|t|今年
every day|t|毎日
every morning|t|毎朝
at once|adv|すぐに
right now|t|今すぐ
of course|adv|もちろん
a lot|adv|たくさん
thank you|int|ありがとう
oh|int|
well,|int|
okay|int|
hello|int|
become|v|なる|g|became|become
gain|v|得る|i
earn|v|稼ぐ|g
receive|v|受け取る|g
throw|v|投げる|i|threw|thrown
obey|v|従う|g
care|v|気にする|s
regret|v|後悔する|s
oppose|v|反対する|s
achieve|v|達成する|s
reply|v|返事する|s
prefer|v|好む|g
drop|v|落とす|g
bother|v|悩ます|g
earn|v|稼ぐ|g
shout|v|叫ぶ|g
result|v|結果になる|g
jump|v|飛ぶ|g
present|v|贈る|g
recognize|v|分かる|g
rescue|v|救う|g
fill|v|満たす|g
settle|v|解決する|s
apply|v|当てはまる|g
slip|v|滑る|g
destroy|v|破壊する|s
demand|v|要求する|s
insist|v|主張する|s
admire|v|感心する|s
pick|v|選ぶ|g
raise|v|上げる|i
lie|v|横になる|g|lay|lain
lay|v|置く|g|laid|laid
rest|v|休む|g
exercise|v|運動する|s
last|v|続く|g
deal|v|扱う|g|dealt|dealt
set|v|置く|g|set|set
surprise|v|驚かす|g
gamble|v|賭ける|i
brag|v|自慢する|s
absent|v|欠席する|s
acknowledge|v|認める|i
cooperate|v|協力する|s
refer|v|言及する|s
return|v|帰る|g
sunday|n|日曜日|t
monday|n|月曜日|t
tuesday|n|火曜日|t
wednesday|n|水曜日|t
thursday|n|木曜日|t
friday|n|金曜日|t
saturday|n|土曜日|t
august|n|８月|t
noon|n|正午|t
fool|n|ばか|h
account|n|口座|t
toilet|n|トイレ|p
pity|n|残念|t
hill|n|丘|p
meal|n|食事|t
meals|n|食事|t
program|n|番組|t
counter|n|カウンター|t
deal|n|取引|t
couple|n|夫婦|h
guitar|n|ギター|t
behavior|n|行動|t
tower|n|塔|p
debt|n|借金|t
tears|n|涙|t
gesture|n|身振り|t
members|n|メンバー|h
member|n|メンバー|h
ken|n|ケン|h
law|n|法律|t
coward|n|臆病者|h
gambling|n|ギャンブル|t
dessert|n|デザート|t
argument|n|議論|t
color|n|色|t
narita|n|成田|p
exercise|n|運動|t
dollar|n|ドル|t
television|n|テレビ|t
purse|n|財布|t
chicken|n|鶏肉|t
island|n|島|p
savings|n|貯金|t
sight|n|光景|t
task|n|仕事|t
designer|n|デザイナー|h
situation|n|状況|t
luggage|n|荷物|t
boat|n|ボート|t
bill|n|請求書|t
nonsense|n|ばかげたこと|t
model|n|模型|t
clouds|n|雲|t
position|n|地位|t
present|n|贈り物|t
pain|n|痛み|t
koala|n|コアラ|a
pleasure|n|喜び|t
piece|n|一片|t
favor|n|お願い|t
air|n|空気|t
ambulance|n|救急車|t
cookies|n|クッキー|t
stars|n|星|t
sort|n|種類|t
statement|n|声明|t
vegetables|n|野菜|t
whisky|n|ウイスキー|t
hobby|n|趣味|t
diamond|n|ダイヤモンド|t
studio|n|スタジオ|p
thief|n|泥棒|h
swahili|n|スワヒリ語|t
ski|n|スキー|t
customers|n|客|h
fellow|n|やつ|h
ring|n|指輪|t
tune|n|曲|t
pizza|n|ピザ|t
total|n|合計|t
stranger|n|見知らぬ人|h
beck|n|ベック|h
date|n|日付|t
sports|n|スポーツ|t
parts|n|部品|t
branches|n|枝|t
desk|n|机|t
singer|n|歌手|h
image|n|イメージ|t
spirit|n|精神|t
extent|n|程度|t
admiration|n|称賛|t
wire|n|電線|t
extension|n|延長|t
cooperation|n|協力|t
oranges|n|オレンジ|t
seasons|n|季節|t
shock|n|ショック|t
shuttle|n|シャトル|t
boston|n|ボストン|p
rest|n|残り|t
rest|n|休息|t
walk|n|散歩|t
watch|n|腕時計|t
hope|n|希望|t
talk|n|話|t
help|n|助け|t
change|n|変更|t
study|n|研究|t
surprise|n|驚き|t
sympathy|n|同情|t
better|a|より良い
best|a|一番良い
lovely|a|素敵な
dead|fx|死んでいる
bare|a|裸の
impossible|na|不可能
thick|a|厚い
idle|na|怠惰
extraordinary|na|並外れた
sympathetic|na|同情的
older|a|年上の
pleased|fx|喜んでいる
pale|a|青白い
lower|a|低い
political|na|政治的
prettier|a|よりきれいな
windy|a|風の強い
satisfied|fx|満足している
surprised|fx|驚いている
whole|na|全体
direct|na|直接
definite|na|明確
obliged|fx|感謝している
spacious|a|広い
comfortable|na|快適
blind|na|目の見えない
third|a|三番目の
burning|fx|燃えている
senior|na|年上
liberal|na|寛大
entire|na|全体
several|a|いくつかの
few|a|少しの
filled|fx|いっぱいだ
sorry|na|残念
somewhere|adv|どこかに
nearly|adv|ほとんど
clearly|adv|明らかに
indeed|adv|本当に
later|adv|後で
hardly|adv|ほとんど～ない
deeply|adv|深く
exactly|adv|正確に
entirely|adv|完全に
forward|adv|前へ
someday|t|いつか
till|p|まで
above|p|の上に
o|int|
'clock|int|
''|int|
good-bye|int|さようなら
address|n|住所|t
affinity|n|類似|t
amy|n|エイミー|h
takeshi|n|タケシ|h
smith|n|スミス|h
mr smith|n|スミスさん|h
appointment|n|約束|t
area|n|分野|t
arrival|n|到着|t
baggage|n|荷物|t
bees|n|蜂|a
bottom|n|底|t
breath|n|息|t
bucket|n|バケツ|t
buckets|n|バケツ|t
butter|n|バター|t
canada|n|カナダ|p
cherry|n|桜|t
cherry tree|n|桜の木|t
church|n|教会|p
coins|n|硬貨|t
college|n|大学|p
composition|n|作文|t
air conditioning|n|エアコン|t
crowd|n|人混み|t
dad|n|お父さん|h
departure|n|出発|t
details|n|詳細|t
detective|n|探偵|h
difference|n|違い|t
difficulty|n|困難|t
disney|n|ディズニー|t
dumps|n|落ち込み|t
duty|n|勤務|t
ease|n|容易さ|t
enemies|n|敵|h
excess|n|過度|t
express|n|速達|t
fare|n|運賃|t
foot|n|足|t
force|n|力|t
freedom|n|自由|t
genius|n|天才|t
ghost|n|幽霊|t
heat|n|暑さ|t
ice|n|氷|t
instant|n|瞬間|t
jacket|n|上着|t
kanji|n|漢字|t
kinds|n|種類|t
lack|n|不足|t
ladder|n|はしご|t
lawyer|n|弁護士|h
leg|n|足|t
luck|n|運|t
map|n|地図|t
math|n|数学|t
movement|n|運動|t
mystery|n|推理|t
mystery story|n|推理小説|t
none|n|何も|t
obligation|n|義務|t
occasion|n|機会|t
occupation|n|職業|t
opportunity|n|機会|t
others|n|他人|h
outside|n|外|t
policeman|n|警官|h
population|n|人口|t
proposal|n|提案|t
purpose|n|目的|t
roof|n|屋根|t
sand|n|砂|t
satisfaction|n|満足|t
service|n|勤務|t
shellfish|n|貝|t
sides|n|側|t
statesman|n|政治家|h
stewardess|n|スチュワーデス|h
straw|n|わら|t
team|n|チーム|t
textbooks|n|教科書|t
theaters|n|映画館|p
typist|n|タイピスト|h
waste|n|無駄|t
worst|n|最悪|t
stick|n|棒|t
cash|n|現金|t
aside|adv|脇に
abrupt|na|突然
absolutely|adv|絶対に
alive|fx|生きている
anxious|na|心配
astonished|fx|驚いている
acquainted|fx|知り合いだ
careless|na|不注意
certain|na|確か
colder|a|より寒い
defeated|fx|負けている
diligent|na|勤勉
due|na|予定
engaged|fx|婚約している
exhausted|fx|疲れきっている
familiar|na|馴染み
fatter|a|より太い
foreign|fx|外国の
gladly|adv|喜んで
gradually|adv|徐々に
handy|na|便利
happiest|na|一番幸せ
humid|a|蒸し暑い
immediately|adv|すぐに
immoral|na|不道徳
independent|fx|独立している
injured|fx|怪我をしている
intelligible|fx|理解できる
likely|na|ありそう
longer|a|より長い
mere|a|単なる
possible|na|可能
precious|na|貴重
probable|na|ありそう
proud|fx|誇りに思っている
quick|a|速い
recent|fx|最近の
sleepy|a|眠い
slightly|adv|少し
sore|fx|怒っている
special|na|特別
sufficiently|adv|十分に
supposed|na|想定
terribly|adv|ひどく
transient|na|一時的
unable|fx|できない
unusual|a|珍しい
upstairs|adv|二階に
usual|na|普通
valuable|na|貴重
wise|a|賢い
worth|fx|価値がある
badly|adv|下手に
finally|adv|ついに
financially|adv|経済的に
rather|adv|かなり
twice|adv|二度
instead|adv|代わりに
either|adv|も
less|adv|より少なく
off|adv|離れて
born|fx|生まれた
ten-year-old|fx|十歳の
on foot|adv|徒歩で
in advance|adv|前もって
as soon as possible|adv|できるだけ早く
on occasion|adv|時々
on duty|adv|勤務中で
with ease|adv|簡単に
to excess|adv|過度に
accustom|v|慣れる|i
acquaint|v|知らせる|i
acquit|v|無罪にする|s|acquitted|acquitted
add|v|加える|i
adopt|v|採用する|s
avail|v|利用する|s
bear|v|持つ|g|bore|borne
betray|v|裏切る|g
boast|v|自慢する|s
boot|v|追い出す|g
bind|v|縛る|g|bound|bound
chance|v|偶然する|s
clear|v|晴れる|i
commute|v|通勤する|s
compare|v|比べる|i
compensate|v|補償する|s
contrive|v|なんとかする|s
cotton|v|好きになる|g|cottoned|cottoned
count|v|数える|i
dare|v|あえてする|s
defeat|v|負かす|g
dispose|v|処分する|s
divide|v|分かれる|i
engage|v|約束する|s
exhaust|v|使い果たす|g
express|v|表す|g
face|v|向かい合う|g
fix|v|直す|g
gear|v|準備する|s
occur|v|起こる|g
praise|v|褒める|i
prepare|v|準備する|s
recollect|v|思い出す|g
refuse|v|断る|g
rely|v|頼る|g
rent|v|借りる|i
retort|v|言い返す|g
rise|v|昇る|g|rose|risen
rotate|v|回転する|s
shed|v|流す|g|shed|shed
shake|v|振る|g|shook|shaken
shut|v|閉まる|g|shut|shut
soothe|v|なだめる|i
strike|v|打つ|g|struck|struck
term|v|呼ぶ|g
waste|v|無駄にする|s
wonder|v|思う|g
stick|v|固執する|s
;|int|
"""

# ============================================================================
# 2b. What one does with a noun: the noun's use-verbs, keyed by the English light verb
#     (Mel'cuk's lexical functions).  noun | light verbs (* = take/have/get/make) | ja | conj | particle
# ============================================================================
USE_TEXT = """
bath|take,have|入る|g|に
shower|take,have|浴びる|i|を
train|take,catch,ride,get|乗る|g|に
bus|take,catch,ride,get|乗る|g|に
taxi|take,catch,ride,get|乗る|g|に
plane|take,catch,ride,get|乗る|g|に
airplane|take,catch,ride,get|乗る|g|に
bicycle|ride,take|乗る|g|に
horse|ride|乗る|g|に
medicine|take|飲む|g|を
photo|take|撮る|g|を
picture|take|撮る|g|を
walk|take,have,go|散歩する|s|を
rest|take,have|休む|g|-
nap|take,have|昼寝する|s|を
trip|take,make|旅行する|s|を
exam|take|受ける|i|を
examination|take|受ける|i|を
test|take|受ける|i|を
seat|take,have|座る|g|に
look|take,have|見る|i|を
care|take|気をつける|i|に
part|take|参加する|s|に
cold|have|ひく|g|を|stative
cold|catch,get|ひく|g|を
fever|have|ある|aru|が
headache|have|する|s|が
dream|have|見る|i|を
lunch|have,eat|食べる|i|を
dinner|have,eat|食べる|i|を
breakfast|have,eat|食べる|i|を
party|have,hold,give,throw|開く|g|を
meeting|have,hold|開く|g|を
accident|have|遭う|g|に
time|have|ある|aru|が
money|have|ある|aru|が
chance|have,get|ある|aru|が
idea|have|ある|aru|が
class|have|ある|aru|が
lesson|have|ある|aru|が
trouble|have|困る|g|に
fun|have|楽しむ|g|を
mistake|make|する|s|を
mistakes|make|する|s|を
decision|make|する|s|を
speech|make,give|する|s|を
promise|make|する|s|を
noise|make|立てる|i|を
friends|make|作る|g|を
plan|make|立てる|i|を
effort|make|する|s|を
efforts|make|する|s|を
progress|make|する|s|を
call|make,give|かける|i|を
ring|give|かける|i|を
answer|give,make|する|s|を
advice|give|する|s|を
question|ask|する|s|を
favor|ask,do|する|s|を
homework|do|する|s|を
work|do|する|s|を
business|do|する|s|を
best|do|尽くす|g|を
dishes|do|洗う|g|を
guitar|play|弾く|g|を
piano|play|弾く|g|を
violin|play|弾く|g|を
trumpet|play|吹く|g|を
flute|play|吹く|g|を
drums|play|叩く|g|を
tennis|play|する|s|を
soccer|play|する|s|を
baseball|play|する|s|を
cards|play|する|s|を
game|play|する|s|を
chess|play|する|s|を
lie|tell|つく|g|を
truth|tell|言う|g|を
story|tell|する|s|を
promise|keep|守る|g|を
promise|break|破る|g|を
law|break|破る|g|を
law|obey|守る|g|を
diary|keep|つける|i|を
secret|keep|守る|g|を
dog|keep,have|飼う|g|を|stative
cat|keep,have|飼う|g|を|stative
window|break|割る|g|を
glass|break|割る|g|を
record|break|破る|g|を
heart|break|傷つける|i|を
weight|lose,gain|減る|g|が
way|lose|迷う|g|に
temper|lose|怒る|g|を
game|lose|負ける|i|に
game|win|勝つ|g|に
race|win|勝つ|g|に
race|lose|負ける|i|に
prize|win,get|もらう|g|を
attention|pay|払う|g|を
visit|pay,make|する|s|を
fish|catch|釣る|g|を
money|spend|使う|g|を
money|make,earn|稼ぐ|g|を
job|get,find|見つける|i|を
job|lose|失う|g|を
letter|get,receive|もらう|g|を
present|get,receive|もらう|g|を
fire|set|つける|i|を
table|set|整える|i|を
alarm|set|かける|i|を
risk|run,take|冒す|g|を
hello|say|言う|g|を
prayer|say|祈る|g|を
shopping|go|行く|g|に
fishing|go|行く|g|に
skiing|go|行く|g|に
swimming|go|行く|g|に
"""
USE = {}          # noun -> {light verb: (ja, conj, particle)}
for line in USE_TEXT.strip().splitlines():
    f = line.split("|")
    noun, verbs, ja, conj, part = f[:5]
    for v in verbs.split(","):
        USE.setdefault(noun, {})[v] = (ja, conj, part, len(f) > 5)
LIGHT = {v for d in USE.values() for v in d}

# ============================================================================
# 3. English morphology I know, used to look words up by lemma
# ============================================================================
LEX = {}          # word -> list of entries (pos, ja, extra...)
VERB_FORMS = {}   # surface -> (lemma, form) form in base/3sg/past/ing/pp
def add(word, entry):
    LEX.setdefault(word, []).append(entry)

def regular_forms(base):
    if base.endswith("e") and not base.endswith("ee"):
        past, ing = base + "d", base[:-1] + "ing"
    elif base.endswith("y") and base[-2] not in "aeiou":
        past, ing = base[:-1] + "ied", base + "ing"
    elif re.search(r"[^aeiou][aeiou][bdglmnprt]$", base) and base not in ("open", "visit", "listen", "happen", "enter", "answer", "remember", "order", "offer"):
        past, ing = base + base[-1] + "ed", base + base[-1] + "ing"
    else:
        past, ing = base + "ed", base + "ing"
    if base.endswith(("s", "sh", "ch", "x", "o")): s3 = base + "es"
    elif base.endswith("y") and base[-2] not in "aeiou": s3 = base[:-1] + "ies"
    else: s3 = base + "s"
    return s3, past, ing

for line in LEXICON_TEXT.strip().splitlines():
    f = line.split("|")
    word, pos, ja = f[0], f[1], f[2]
    if pos == "v":
        conj = f[3]
        s3, past, ing = regular_forms(word)
        if len(f) > 4: past = f[4]
        pp = f[5] if len(f) > 5 else past
        add(word, ("v", ja, conj))
        for surface, form in ((word, "base"), (s3, "3sg"), (past, "past"), (ing, "ing"), (pp, "pp")):
            VERB_FORMS.setdefault(surface, []).append((word, form))
    elif pos == "n":
        add(word, ("n", ja, f[3]))
        if not word.endswith("s") and " " not in word:
            pl = word + ("es" if word.endswith(("s", "sh", "ch", "x")) else "s")
            if pl not in LEX: add(pl, ("n", ja, f[3]))
    else:
        add(word, (pos, ja))
# irregular English verb forms the regular rules get wrong
for surface, lemma, form in (("has", "have", "3sg"), ("does", "do", "3sg"), ("goes", "go", "3sg"),
                             ("am", "be", "base"), ("is", "be", "3sg"), ("are", "be", "base"), ("were", "be", "past"),
                             ("being", "be", "ing"), ("having", "have", "ing"), ("running", "run", "ing"),
                             ("swimming", "swim", "ing"), ("sitting", "sit", "ing"), ("getting", "get", "ing"),
                             ("cutting", "cut", "ing"), ("putting", "put", "ing"), ("hitting", "hit", "ing"),
                             ("beginning", "begin", "ing"), ("stopping", "stop", "ing"), ("planning", "plan", "ing"),
                             ("shopping", "shop", "ing"), ("dying", "die", "ing"), ("lying", "lie", "ing")):
    VERB_FORMS.setdefault(surface, []).append((lemma, form))

PRONOUNS = {  # surface -> (japanese, role: s subject / o object / p possessive / r reflexive, person)
    "i": ("私", "s"), "me": ("私", "o"), "my": ("私の", "p"), "mine": ("私のもの", "o"), "myself": ("自分", "r"),
    "you": ("あなた", "so"), "your": ("あなたの", "p"), "yours": ("あなたのもの", "o"), "yourself": ("自分", "r"),
    "he": ("彼", "s"), "him": ("彼", "o"), "his": ("彼の", "p"), "himself": ("自分", "r"),
    "she": ("彼女", "s"), "her": ("彼女", "op"), "hers": ("彼女のもの", "o"), "herself": ("自分", "r"),
    "we": ("私たち", "s"), "us": ("私たち", "o"), "our": ("私たちの", "p"), "ours": ("私たちのもの", "o"), "ourselves": ("自分たち", "r"),
    "they": ("彼ら", "s"), "them": ("彼ら", "o"), "their": ("彼らの", "p"), "theirs": ("彼らのもの", "o"), "themselves": ("自分たち", "r"),
    "it": ("それ", "so"), "its": ("その", "p"),
    "everyone": ("みんな", "so"), "everybody": ("みんな", "so"), "someone": ("誰か", "so"), "somebody": ("誰か", "so"),
    "anyone": ("誰か", "so"), "anybody": ("誰か", "so"), "nobody": ("誰も", "so"),
    "tom": ("トム", "so"), "john": ("ジョン", "so"), "mary": ("メアリー", "so"), "jim": ("ジム", "so"),
    "bob": ("ボブ", "so"), "jane": ("ジェーン", "so"), "tony": ("トニー", "so"), "emi": ("エミ", "so"), "joan": ("ジョアン", "so"),
    "this": ("これ", "so"), "that": ("それ", "so"), "these": ("これら", "so"), "those": ("それら", "so"),
    "something": ("何か", "so"), "nothing": ("何も", "so"), "everything": ("全て", "so"), "anything": ("何か", "so"),
    "one": ("一つ", "o"), "whoever": ("誰でも", "s"),
}
DETS = {"the": "", "a": "", "an": "", "this": "この", "that": "その", "these": "これらの", "those": "それらの",
        "some": "", "any": "", "every": "毎", "each": "それぞれの", "another": "別の", "no": "", "such": "そんな",
        "all": "全ての", "many": "たくさんの", "most": "ほとんどの", "own": "自分の", "last": "最後の", "next": "次の",
        "other": "他の", "both": "両方の", "much": "たくさんの", "plenty": "たくさんの", "whole": "全体の"}
PREPS = {"in": ("で", "に"), "at": ("で", "に"), "on": ("で", "に"), "to": ("に", "に"), "into": ("に", "に"),
         "from": ("から", "から"), "with": ("と", "と"), "for": ("のために", "のために"), "by": ("で", "のそばに"),
         "about": ("について", "について"), "near": ("の近くで", "の近くに"), "under": ("の下で", "の下に"),
         "over": ("の上で", "の上に"), "after": ("の後で", "の後に"), "before": ("の前に", "の前に"),
         "without": ("なしで", "なしで"), "through": ("を通って", "を通って"), "across": ("を渡って", "を渡って"),
         "along": ("に沿って", "に沿って"), "behind": ("の後ろで", "の後ろに"), "between": ("の間で", "の間に"),
         "among": ("の間で", "の間に"), "around": ("の周りで", "の周りに"), "against": ("に反対して", "に反対して"),
         "during": ("の間", "の間"), "until": ("まで", "まで"), "since": ("から", "から"), "of": ("の", "の"),
         "like": ("のように", "のように"), "as": ("として", "として"), "than": ("より", "より"),
         "above": ("の上で", "の上に"), "till": ("まで", "まで"), "outside": ("の外で", "の外に"), "inside": ("の中で", "の中に"),
         "beside": ("のそばで", "のそばに"), "beyond": ("の向こうに", "の向こうに"), "except": ("以外", "以外"),
         "despite": ("にもかかわらず", "にもかかわらず"), "within": ("以内に", "以内に"), "upon": ("の上に", "の上に"),
         "onto": ("の上に", "の上に"), "towards": ("の方へ", "の方へ"), "toward": ("の方へ", "の方へ"),
         "considering": ("を考えると", "を考えると"), "around": ("ごろ", "ごろ")}
PHRASAL = {("get", "on"): ("乗る", "g"), ("get", "off"): ("降りる", "i"), ("get", "up"): ("起きる", "i"),
           ("get", "to"): ("着く", "g"), ("get", "along"): ("うまくやる", "g"), ("get", "back"): ("戻る", "g"),
           ("look", "for"): ("探す", "g"), ("look", "at"): ("見る", "i"), ("look", "after"): ("世話をする", "s"),
           ("put", "on"): ("着る", "i"), ("take", "off"): ("脱ぐ", "g"), ("pick", "up"): ("拾う", "g"),
           ("give", "up"): ("諦める", "i"), ("go", "on"): ("続ける", "i"), ("come", "back"): ("戻る", "g"),
           ("go", "back"): ("戻る", "g"), ("wait", "for"): ("待つ", "g"), ("listen", "to"): ("聞く", "g"),
           ("find", "out"): ("見つけ出す", "g"), ("turn", "on"): ("つける", "i"), ("turn", "off"): ("消す", "g"),
           ("grow", "up"): ("育つ", "g"), ("come", "in"): ("入る", "g"), ("go", "out"): ("出かける", "i"),
           ("sit", "down"): ("座る", "g"), ("stand", "up"): ("立つ", "g"), ("run", "away"): ("逃げる", "i"),
           ("call", "on"): ("訪ねる", "i"), ("come", "home"): ("帰る", "g"), ("go", "home"): ("帰る", "g"),
           ("wake", "up"): ("起きる", "i"), ("put", "off"): ("延期する", "s"), ("set", "out"): ("出発する", "s"),
           ("take", "care"): ("気をつける", "i"), ("come", "across"): ("出くわす", "g"), ("fall", "asleep"): ("寝入る", "g"),
           ("hear", "from"): ("便りがある", "g"), ("belong", "to"): ("属する", "s"), ("depend", "on"): ("頼る", "g"),
           ("agree", "with"): ("賛成する", "s"), ("get", "married"): ("結婚する", "s"), ("take", "place"): ("起こる", "g"),
           ("break", "down"): ("故障する", "s"), ("get", "well"): ("良くなる", "g"), ("go", "fishing"): ("釣りに行く", "g"),
           ("go", "shopping"): ("買い物に行く", "g"), ("come", "true"): ("実現する", "s"), ("keep", "on"): ("続ける", "i"),
           ("go", "away"): ("立ち去る", "g"), ("get", "lost"): ("道に迷う", "g"), ("take", "part"): ("参加する", "s")}
PARTICLE_AFTER = {("have", "on"): ("着ている", "fixed"), ("put", "on"): ("着る", "i"), ("take", "off"): ("脱ぐ", "g"),
                  ("turn", "on"): ("つける", "i"), ("turn", "off"): ("消す", "g"), ("pick", "up"): ("拾う", "g"),
                  ("give", "up"): ("諦める", "i"), ("put", "off"): ("延期する", "s"), ("call", "up"): ("電話する", "s"),
                  ("wake", "up"): ("起こす", "g"), ("bring", "up"): ("育てる", "i"), ("throw", "away"): ("捨てる", "i"),
                  ("put", "away"): ("片付ける", "i"), ("look", "up"): ("調べる", "i"), ("try", "on"): ("試着する", "s"),
                  ("write", "down"): ("書き留める", "i"), ("put", "down"): ("置く", "g"), ("let", "in"): ("入れる", "i"),
                  ("count", "in"): ("数に入れる", "i"), ("fix", "up"): ("用意する", "s"), ("see", "off"): ("見送る", "g"),
                  ("take", "back"): ("取り戻す", "g"), ("give", "back"): ("返す", "g"), ("send", "back"): ("送り返す", "g"),
                  ("cut", "down"): ("切り倒す", "g"), ("carry", "out"): ("実行する", "s"), ("find", "out"): ("見つけ出す", "g"),
                  ("make", "up"): ("作り上げる", "i"), ("turn", "down"): ("断る", "g"), ("hand", "in"): ("提出する", "s"),
                  ("keep", "up"): ("続ける", "i")}
VERB_IDIOM = {("get", ("in", "touch", "with")): ("連絡を取る", "g", "と"), ("look", ("forward", "to")): ("楽しみにする", "s", "を"),
              ("take", ("care", "of")): ("世話をする", "s", "の"), ("get", ("rid", "of")): ("取り除く", "g", "を"),
              ("make", ("fun", "of")): ("からかう", "g", "を"), ("keep", ("up", "with")): ("ついていく", "g", "に"),
              ("put", ("up", "with")): ("我慢する", "s", "を"), ("run", ("out", "of")): ("切らす", "g", "を"),
              ("get", ("along", "with")): ("うまくやる", "g", "と"), ("make", ("use", "of")): ("利用する", "s", "を"),
              ("pay", ("attention", "to")): ("注意を払う", "g", "に"), ("catch", ("up", "with")): ("追いつく", "g", "に"),
              ("take", ("part", "in")): ("参加する", "s", "に"), ("look", ("after",)): ("世話をする", "s", "の"),
              ("get", ("over",)): ("乗り越える", "i", "を"), ("come", ("across",)): ("出くわす", "g", "に"),
              ("look", ("into",)): ("調べる", "i", "を"), ("make", ("up", "for")): ("埋め合わせる", "i", "を"),
              ("think", ("of",)): ("思いつく", "g", "を"), ("speak", ("ill", "of")): ("悪く言う", "g", "を"),
              ("get", ("to",)): ("着く", "g", "に"), ("hear", ("from",)): ("便りがある", "aru", "から"),
              ("laugh", ("at",)): ("笑う", "g", "を"), ("wait", ("for",)): ("待つ", "g", "を"), ("look", ("for",)): ("探す", "g", "を"),
              ("look", ("at",)): ("見る", "i", "を"), ("listen", ("to",)): ("聞く", "g", "を"), ("ask", ("for",)): ("求める", "i", "を"),
              ("care", ("for",)): ("好む", "g", "を"), ("agree", ("with",)): ("賛成する", "s", "に"), ("agree", ("to",)): ("同意する", "s", "に"),
              ("belong", ("to",)): ("属する", "s", "に"), ("depend", ("on",)): ("頼る", "g", "に"), ("rely", ("on",)): ("頼る", "g", "に"),
              ("insist", ("on",)): ("主張する", "s", "を"), ("apologize", ("for",)): ("謝る", "g", "を"), ("worry", ("about",)): ("心配する", "s", "を"),
              ("think", ("about",)): ("考える", "i", "を"), ("talk", ("about",)): ("話す", "g", "について"), ("know", ("about",)): ("知っている", "fixed", "について"),
              ("arrive", ("at",)): ("着く", "g", "に"), ("arrive", ("in",)): ("着く", "g", "に"), ("get", ("married", "to")): ("結婚する", "s", "と"),
              ("succeed", ("in",)): ("成功する", "s", "に"), ("fail", ("in",)): ("失敗する", "s", "に"), ("boast", ("of",)): ("自慢する", "s", "を"),
              ("dispose", ("of",)): ("処分する", "s", "を"), ("approve", ("of",)): ("賛成する", "s", "に"), ("stick", ("to",)): ("固執する", "s", "に"),
              ("deal", ("with",)): ("扱う", "g", "を"), ("die", ("of",)): ("死ぬ", "g", "で"), ("suffer", ("from",)): ("苦しむ", "g", "に"),
              ("complain", ("of",)): ("不平を言う", "g", "について"), ("search", ("for",)): ("探す", "g", "を"), ("account", ("for",)): ("説明する", "s", "を")}
ADJ_PREP = {("fond", "of"): ("好き", "na", "が"), ("afraid", "of"): ("怖い", "a", "が"), ("interested", "in"): ("興味がある", "fx", "に"),
            ("good", "at"): ("得意", "na", "が"), ("poor", "at"): ("苦手", "na", "が"), ("late", "for"): ("遅れる", "v", "に"),
            ("proud", "of"): ("誇りに思っている", "fx", "を"), ("tired", "of"): ("飽きている", "fx", "に"), ("used", "to"): ("慣れている", "fx", "に"),
            ("absent", "from"): ("欠席する", "v", "を"), ("full", "of"): ("いっぱいだ", "fx", "で"), ("famous", "for"): ("有名", "na", "で"),
            ("different", "from"): ("違う", "v", "と"), ("busy", "with"): ("忙しい", "a", "で"), ("sure", "of"): ("確信している", "fx", "を"),
            ("satisfied", "with"): ("満足している", "fx", "に"), ("angry", "with"): ("怒っている", "fx", "に"), ("angry", "at"): ("怒っている", "fx", "に"),
            ("familiar", "with"): ("詳しい", "a", "に"), ("familiar", "to"): ("馴染みがある", "fx", "に"), ("independent", "of"): ("独立している", "fx", "から"),
            ("free", "from"): ("ない", "fx", "が"), ("ready", "for"): ("用意ができている", "fx", "の"), ("worried", "about"): ("心配している", "fx", "を"),
            ("anxious", "about"): ("心配している", "fx", "を"), ("careful", "of"): ("気をつける", "v", "に"), ("aware", "of"): ("気づいている", "fx", "に"),
            ("capable", "of"): ("できる", "fx", "が"), ("responsible", "for"): ("責任がある", "fx", "に"), ("similar", "to"): ("似ている", "fx", "に"),
            ("kind", "to"): ("親切", "na", "に"), ("married", "to"): ("結婚している", "fx", "と"), ("pleased", "with"): ("喜んでいる", "fx", "を"),
            ("surprised", "at"): ("驚いている", "fx", "に"), ("sorry", "for"): ("気の毒に思う", "v", "を"), ("glad", "of"): ("嬉しい", "a", "が"),
            ("dependent", "on"): ("頼っている", "fx", "に"), ("ashamed", "of"): ("恥じている", "fx", "を"), ("fit", "for"): ("向いている", "fx", "に"),
            ("rich", "in"): ("豊か", "na", "が"), ("short", "of"): ("足りない", "fx", "が"), ("sick", "of"): ("うんざりしている", "fx", "に"),
            ("done", "with"): ("済ませている", "fx", "を"), ("gone", "on"): ("ほれこんでいる", "fx", "に"), ("bound", "for"): ("行き", "fx", "")}
WH = {"what": ("何", "o"), "who": ("誰", "s"), "where": ("どこ", "loc"), "when": ("いつ", "t"),
      "why": ("なぜ", "t"), "how": ("どう", "adv"), "which": ("どれ", "o")}
MODALS = {"can": "can", "could": "can", "must": "must", "should": "should", "ought": "should", "may": "may",
          "might": "may", "will": "will", "'ll": "will", "would": "will", "'d": "will", "shall": "will"}
NUMBERS = {"one": "１", "two": "２", "three": "３", "four": "４", "five": "５", "six": "６", "seven": "７",
           "eight": "８", "nine": "９", "ten": "１０", "twenty": "２０", "thirty": "３０", "fifty": "５０",
           "hundred": "１００", "thousand": "１０００", "million": "１００万", "half": "半分", "first": "最初の",
           "fourteen": "１４", "fifteen": "１５", "forty": "４０", "sixty": "６０", "dozen": "１ダースの"}
MULTIWORD = sorted([w for w in LEX if " " in w], key=len, reverse=True)
SUBORD = {"when": "とき", "while": "間", "if": "たら", "because": "ので", "since": "ので", "as": "ので", "unless": "なければ", "until": "まで",
          "as soon as": "とすぐに", "so that": "ように", "though": "が、", "although": "が、", "even though": "が、", "after": "後で", "before": "前に",
          "as if": "かのように", "as though": "かのように", "so long as": "限り", "as long as": "限り", "once": "たら", "even if": "としても",
          "in case": "場合に備えて", "whenever": "ときはいつも", "every time": "たびに", "now that": "ので", "by the time": "までに"}
COORD = {"and": "", "but": "が、", "so": "ので", "or": "か、", "and then": "それから", "while": "が、"}
HEARSAY = {"think": "と思う", "feel": "と感じる", "hear": "と聞いている", "suppose": "と思う", "guess": "と思う", "believe": "と信じている", "hope": "と願う", "know": "と知っている"}

def subord_form(inner, conj):
    """a parsed subordinate clause -> Japanese clause + connective, Japanese-first order"""
    if conj in ("if", "once"): inner.past = True
    if conj == "after": inner.past = True
    if conj == "unless": inner.neg = True
    if conj == "so that" and inner.modal in ("may", "can"): inner.modal = "can"
    if conj == "so that" and inner.modal == "will": inner.modal = None
    body = generate(inner, polite=False, embedded=True)[:-1]
    if conj in ("if", "once"): return body + "ら"
    if conj == "unless": return (body[:-2] + "なければ") if body.endswith("ない") else body + "なら"
    if conj == "while" and inner.verb and not inner.copula and inner.subj is None:
        ipred = predicate_form(inner, False); p = clause_pred(inner)
        if p is not None: return body[:-len(ipred)] + p.masu_stem() + "ながら"
    return body + SUBORD[conj]

def clause_pred(c):
    """the Pred object a clause's predicate is built from, or None for the special cases"""
    if c.copula is not None:
        kind, *rest = c.copula
        if kind == "adj": return adj_pred(rest[0][0], rest[0][1])
        if kind == "np": return Pred("na", rest[0])
        if kind in ("pp", "adv"): return Pred("v1", "い") if c.subj_cls in ("h", "a") else Pred("v5", "ある")
        return None
    if c.verb is None: return None
    lemma, ja, conj = c.verb
    if conj in ("fixed", "ia_fixed") or c.want or c.modal or c.perf or "+" in lemma: return None
    p = make_pred(ja, conj)
    if c.prog or lemma in STATIVE: p = p.progressive()
    return p

def clause_te(c):
    """clause as a te-form chain link: 私は質問があって"""
    body = generate(c, polite=False)[:-1]
    ipred = predicate_form(c, False); p = clause_pred(c)
    if p is None or c.neg or c.past and p.kind in ("ia", "na"):
        return body + "、"
    try:
        return body[:-len(ipred)] + p.te()
    except Exception:
        return body + "、"

DURATION = {"years": "年間", "year": "年間", "months": "ヶ月", "month": "ヶ月", "weeks": "週間", "week": "週間", "days": "日間", "day": "日",
            "hours": "時間", "hour": "時間", "minutes": "分", "minute": "分", "times": "回", "time": "回"}

CONTRACTED = {"don": "do", "doesn": "does", "didn": "did", "isn": "is", "aren": "are", "wasn": "was",
              "weren": "were", "haven": "have", "hasn": "has", "hadn": "had", "won": "will", "wouldn": "would",
              "couldn": "could", "shouldn": "should", "mustn": "must", "oughtn": "ought", "can": "can",
              "needn": "need", "cannot": "can"}
def classify(tok):
    """All readings of a token: list of (class, payload)."""
    if tok in CONTRACTED and tok not in ("can",): tok = CONTRACTED[tok]
    out = []
    if tok in (".", "?", "!"): return [("end", tok)]
    if tok in ("'t", "not"): return [("not", None)]
    if tok == "n't": return [("not", None)]
    if tok in ("'s",): return [("poss", None), ("aux_be3", None)]
    if tok == "'m": return [("aux_be", None)]
    if tok == "'re": return [("aux_be", None)]
    if tok == "'ve": return [("aux_have", None)]
    if tok in ("do", "does", "did"): out.append(("aux_do", "past" if tok == "did" else "pres"))
    if tok in ("am", "is", "are", "was", "were", "be", "been", "being"):
        out.append(("aux_be", "past" if tok in ("was", "were") else "pres"))
    if tok in ("have", "has", "had"): out.append(("aux_have", "past" if tok == "had" else "pres"))
    if tok in MODALS: out.append(("modal", (MODALS[tok], tok in ("could", "would", "might", "'d"))))
    if tok == "to": out.append(("to", None))
    if tok == "that": out.append(("that", None))
    if tok == "there": out.append(("there", None))
    if tok == "let": out.append(("let", None))
    if tok == "please": out.append(("please", None))
    if tok in ("yes", "no") : out.append(("yesno", "はい" if tok == "yes" else "いいえ"))
    if tok in ("and", "or", "but", "because", "if", "when", "so", "while", "as", "though", "although", "whether", "unless", "until", "once", "since", "after", "before"):
        out.append(("conj", tok))
    if tok in WH: out.append(("wh", WH[tok]))
    if tok in PRONOUNS: out.append(("pron", PRONOUNS[tok]))
    if tok in DETS: out.append(("det", DETS[tok]))
    if tok in NUMBERS: out.append(("num", NUMBERS[tok]))
    if re.fullmatch(r"\d+", tok): out.append(("num", tok.translate(str.maketrans("0123456789", "０１２３４５６７８９"))))
    if tok in PREPS: out.append(("prep", PREPS[tok]))
    for lemma, form in VERB_FORMS.get(tok, []):
        for e in LEX[lemma]:
            if e[0] == "v": out.append(("verb", (lemma, form, e[1], e[2])))
    for e in LEX.get(tok, []):
        if e[0] == "n": out.append(("noun", (e[1], e[2])))
        elif e[0] in ("a", "na", "fx"): out.append(("adj", (e[1], e[0])))
        elif e[0] == "adv": out.append(("adv", e[1]))
        elif e[0] == "t": out.append(("time", e[1]))
        elif e[0] == "int": out.append(("int", e[1]))
        elif e[0] == "conj": out.append(("conj", tok))
        elif e[0] == "det": out.append(("det", e[1]))
    if not any(c == "adj" for c, _ in out) and tok.endswith("er") and len(tok) > 4:
        for base in (tok[:-2], tok[:-1], tok[:-3] + "y", tok[:-3]):
            es = [e for e in LEX.get(base, []) if e[0] in ("a", "na")]
            if es:
                out.append(("adj", ("より" + es[0][1], es[0][0]))); out.append(("adv", "より" + (es[0][1][:-1] + "く" if es[0][0] == "a" and es[0][1].endswith("い") else es[0][1] + "に"))); break
    return out

# ============================================================================
# 4. The frame parser: a DFA over (phase, class) with a few register predicates
# ============================================================================
class ParseFail(Exception):
    pass

class Clause:
    def __init__(self):
        self.subj = None; self.subj_cls = "t"; self.subj_poss = ""; self.subj_adj = ""; self.subj_det = ""
        self.obj = None; self.obj_poss = ""; self.obj_adj = ""; self.obj_det = ""
        self.iobj = None
        self.pp = []               # list of (prep_entry, np string)
        self.time = ""; self.adv = ""
        self.verb = None           # (lemma, ja, conj)
        self.past = False; self.neg = False; self.modal = None; self.prog = False; self.perf = False
        self.copula = None         # ("adj", (ja, kind)) | ("np", string, cls) | ("pp", prep, np)
        self.there = False; self.question = False; self.wh = None; self.imp = False; self.please = False
        self.lets = False; self.want = None; self.degree = ""; self.causative = False
        self.purpose = None; self.purpose_clause = None; self.control = None; self.relative = False
        self.obj_particle = None; self.infinitive = None; self.pre_clause = ""; self.howabout = None; self.exclaim = None; self.wh_cls = "t"
        self.embedded = None       # another Clause (that-clause / infinitive complement)
        self.compare = None        # ("as"|"than", np) degree construction
        self.manner = None         # V-ing clause: stood looking at ... -> 〜ながら
        self.tag = False           # tag question: ..., isn't it ?
        self.sub = False           # a subordinate clause being parsed: stop at a comma or a conjunction
        self.coord = False         # a coordinated clause: keeps は on its subject
        self.suffix = ""           # sentence-final hearsay: ..., i hear -> と聞いている

class Parser:
    def __init__(self, tokens):
        self.toks = tokens; self.i = 0
        self.main = Clause(); self.c = self.main
        self.phase = "PRE"
        self.np = None             # NP under construction: dict(det, poss, adj, num, noun, cls)
        self.np_role = None

    def peek(self, k=0):
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else None
    def readings(self, tok):
        return classify(tok) if tok is not None else []
    def has(self, tok, cls):
        return any(c == cls for c, _ in self.readings(tok))
    def take(self, cls):
        """consume the current token in class cls, return its payload"""
        for c, p in self.readings(self.peek()):
            if c == cls:
                self.i += 1; return p
        raise ParseFail(f"{self.phase}: expected {cls} at '{self.peek()}'")
    def fail(self, why):
        raise ParseFail(f"{self.phase}: {why} at '{self.peek()}'")

    # --- noun phrase: [det|poss|num] [adj]* noun | pron  -> (japanese, class) ---
    def parse_np(self):
        tok = self.peek(); r = dict(self.readings(tok))
        nxt = dict(self.readings(self.peek(1))) if self.peek(1) else {}
        next_is_nominal = "noun" in nxt or "adj" in nxt or "num" in nxt or ("adv" in nxt and self.peek(2) and "adj" in dict(self.readings(self.peek(2))))
        det = poss = num = adjs = ""
        if tok == "a lot" and self.peek(1) == "of": det = "たくさんの"; self.i += 2
        elif tok == "a" and self.peek(1) == "lot" and self.peek(2) == "of": det = "たくさんの"; self.i += 3
        elif tok == "lots" and self.peek(1) == "of": det = "たくさんの"; self.i += 2
        elif "pron" in r and "p" in r["pron"][1] and next_is_nominal:
            poss = r["pron"][0] if r["pron"][1] == "p" else r["pron"][0] + "の"; self.i += 1
        elif "det" in r and (next_is_nominal or "pron" not in r):
            det = r["det"]; self.i += 1
        elif "pron" in r:
            self.i += 1
            ja, role = r["pron"]
            if role == "p": self.fail("possessive pronoun without noun")
            self.last_head = None
            return ja, "h"
        tok = self.peek(); r = dict(self.readings(tok))
        if "num" in r: num = r["num"]; self.i += 1; tok = self.peek(); r = dict(self.readings(tok))
        while True:
            if "adv" in r and self.peek(1) and "adj" in dict(self.readings(self.peek(1))):
                adjs += r["adv"]; self.i += 1; tok = self.peek(); r = dict(self.readings(tok)); continue
            if "adj" in r and self.peek(1) is not None and ("noun" in dict(self.readings(self.peek(1))) or "adj" in dict(self.readings(self.peek(1)))):
                ja, kind = r["adj"]
                adjs += ja + ("な" if kind == "na" and not ja.endswith("な") else ""); self.i += 1
                tok = self.peek(); r = dict(self.readings(tok)); continue
            break
        if "noun" not in r:
            if num and not adjs and not det and not poss:          # a bare number as a noun phrase: "before ten"
                self.last_head = None
                return num + "時" if self.np_role == "pp_time" else num, "t"
            self.fail("noun phrase")
        ja, cls = r["noun"]; self.i += 1
        self.last_head = tok
        # noun-noun compound: english homework, bus stop, road map
        nr2 = dict(self.readings(self.peek())) if self.peek() else {}
        aux_next = self.peek(1) and any(self.has(self.peek(1), k) for k in ("aux_be", "aux_do", "aux_have", "modal"))
        if "noun" in nr2 and not any(k in nr2 for k in ("aux_be", "aux_do", "aux_have", "modal", "prep", "adv", "det", "pron", "adj", "time", "conj", "num")) and ("verb" not in nr2 or aux_next):
            ja2, cls2 = nr2["noun"]; self.i += 1
            self.last_head = self.toks[self.i - 1]
            ja, cls = ja + "の" + ja2, cls2
        # relative clause on the last NP of the sentence: the book (that) i bought / the man who came
        nxt = self.peek(); nr = dict(self.readings(nxt)) if nxt else {}
        rel_pron = nxt in ("who", "which", "that") and self.peek(1) and (self.has(self.peek(1), "verb") or self.has(self.peek(1), "aux_be") or self.has(self.peek(1), "aux_have") or self.has(self.peek(1), "modal"))
        rel_subj = ("pron" in nr and "s" in nr["pron"][1] and self.peek(1) and (self.has(self.peek(1), "verb") or self.has(self.peek(1), "aux_have") or self.has(self.peek(1), "aux_be") or self.has(self.peek(1), "modal")))
        if (rel_pron or rel_subj) and self.rel_ok():
            inner = Clause(); inner.relative = True
            if rel_pron:
                self.i += 1; inner.subj = None; inner.subj_cls = cls
            else:
                sja, scls = self.parse_np(); inner.subj, inner.subj_cls = sja, scls
            saved = self.c; self.c = inner
            self.phase = "VP0"; self.predicate(); self.c = saved
            return generate(inner, polite=False, embedded=True)[:-1] + det + poss + num + adjs + ja, cls
        # possessive 's: John 's car
        if self.peek() == "'s" and self.peek(1) and "noun" in dict(self.readings(self.peek(1))):
            self.i += 1
            inner_ja, cls2 = self.parse_np_rest_after_poss(ja + "の")
            return inner_ja, cls2
        # of-phrase: the top of the mountain -> 山の頂上
        if self.peek() == "of" and self.peek(1) and self.peek(1) not in ("course",):
            self.i += 1
            of_ja, _ = self.parse_np()
            self.last_head = tok
            return det + poss + of_ja + "の" + num + adjs + ja, cls
        return det + poss + num + adjs + ja, cls
    def next_head_uses(self, lemma):
        """does the NP starting at the current token have a head noun with a use-verb for lemma?"""
        j = self.i
        while j < len(self.toks) and self.toks[j] not in (".", "?", "!"):
            r = dict(self.readings(self.toks[j]))
            if "noun" in r and self.toks[j] in USE and lemma in USE[self.toks[j]]: return True
            if "noun" in r or "verb" in r or "prep" in r: return False
            j += 1
        return False

    def rel_ok(self):
        """only attach a relative clause when this NP is the last constituent (no main predicate follows)"""
        return self.c.verb is not None or self.c.copula is not None or self.np_role == "last" or self.phase == "VP0" and self.c.subj is not None

    def parse_np_rest_after_poss(self, prefix):
        ja, cls = self.parse_np()
        return prefix + ja, cls

    def parse(self):
        self.pre()
        if self.phase != "DONE":
            self.fail("unparsed tail")
        return self.main

    def pre(self):
        c = self.c
        while True:
            tok = self.peek(); r = dict(self.readings(tok))
            if tok is None: self.fail("empty")
            if "int" in r or "yesno" in r:
                self.i += 1
                if self.peek() == ",": self.i += 1
                continue
            if "time" in r: c.time += r["time"]; self.i += 1; continue
            if "please" in r: c.please = True; self.i += 1; continue
            if "conj" in r and tok in SUBORD and c is self.main:
                s = self.sub_clause(tok)
                if s is not None: c.pre_clause += s; continue
            if tok == "being" and self.peek(1) and "adj" in dict(self.readings(self.peek(1))) and self.peek(2) == ",":
                ja, kind = dict(self.readings(self.peek(1)))["adj"]; self.i += 3
                c.pre_clause += adj_pred(ja, kind).finite(False, False, False) + "ので"; continue
            if "wh" in r:
                c.question = True; c.wh = r["wh"]; self.i += 1
                nxt = self.peek(); nr = dict(self.readings(nxt)) if nxt else {}
                if tok == "how" and nxt == "about":                                  # how about V-ing / NP ?
                    self.i += 1
                    if self.verb_form(self.peek()) == "ing":
                        inner = Clause(); self.c = inner
                        self.verb_phrase(expect_ing=True, from_copula=True); self.c = c
                        c.howabout = generate(inner, polite=False, embedded=True)[:-1] + "の"
                    else:
                        ja, cls = self.parse_np(); c.howabout = ja
                    return self.end()
                if self.toks[-1] == "!" and (tok == "what" and nxt == "a" or tok == "how" and "adj" in nr and self.peek(1) and self.starts_np(self.peek(1))):
                    # exclamation: what a bad boy he is ! / how beautiful the sun is !
                    c.question = False
                    if tok == "what":
                        ja, cls = self.parse_np(); c.exclaim = ja
                    else:
                        c.exclaim = nr["adj"][0]; self.i += 1
                    if self.peek() and self.starts_np(self.peek()):
                        ja, cls = self.parse_np(); c.subj, c.subj_cls = ja, cls
                        if self.has(self.peek(), "aux_be"): self.i += 1
                    return self.end()
                if tok == "what" and nxt == "time": c.wh = ("何時", "t"); self.i += 1
                elif tok in ("what", "which") and ("noun" in nr or "adj" in nr):          # wh-determiner
                    ja, cls = self.parse_np()
                    c.wh = (("どんな" if tok == "what" else "どの") + ja, "o")
                elif tok == "how" and nxt in ("many", "much") and self.peek(1) and self.starts_np(self.peek(1)):
                    self.i += 1; ja, cls = self.parse_np()
                    c.wh = (ja, "count"); c.wh_cls = cls
                elif tok == "how" and ("adj" in nr or "adv" in nr):                       # how long / how old
                    c.wh = ("何歳" if nxt == "old" else "どのくらい", "adv"); self.i += 1
                if self.peek() == "many" or self.peek() == "much": c.wh = ("いくつ" if self.peek() == "many" else "いくら", "o"); self.i += 1
                return self.after_wh()
            if "there" in r and self.peek(1) and self.has(self.peek(1), "aux_be"):
                self.i += 1; c.there = True
                tense = self.take("aux_be"); c.past = tense == "past"
                if self.has(self.peek(), "not"): self.i += 1; c.neg = True
                ja, cls = self.parse_np(); c.subj, c.subj_cls = ja, cls
                self.tail_pp_adv()
                return self.end()
            if "let" in r and self.peek(1) == "'s":
                self.i += 2; c.lets = True
                return self.verb_phrase(expect_base=True)
            if "let" in r and self.peek(1) in ("me", "us") and self.peek(2) and self.has(self.peek(2), "verb"):
                self.i += 2; c.imp = True; c.please = True; c.causative = True
                return self.verb_phrase(expect_base=True)
            if ("aux_do" in r or "aux_be" in r or "aux_have" in r or "modal" in r) and self.peek(1) and self.starts_np(self.peek(1)):
                return self.question_inversion()
            if "aux_do" in r and self.peek(1) and self.has(self.peek(1), "not"):      # don't V (imperative)
                self.i += 2; c.imp = True; c.neg = True
                return self.verb_phrase(expect_base=True)
            if "verb" in r and not self.starts_np(tok) and any(f == "base" for _, f, _, _ in [p for cc, p in self.readings(tok) if cc == "verb"]) and (self.peek(1) is None or not self.has(self.peek(1), "aux_be")):
                c.imp = True
                return self.verb_phrase(expect_base=True)
            break
        # subject
        ja, cls = self.parse_np(); c.subj, c.subj_cls = ja, cls
        self.phase = "VP0"
        return self.predicate()

    def starts_np(self, tok):
        r = dict(self.readings(tok))
        if "pron" in r or "det" in r or "noun" in r or "num" in r: return True
        if "adj" in r and not ("verb" in r or "adv" in r):                      # bare adjective + noun: sad movies, hot food
            for k in range(4):
                if self.peek(k) == tok:
                    return self.peek(k + 1) is not None and self.has(self.peek(k + 1), "noun") and not self.has(self.peek(k + 1), "verb")
        return False

    def after_wh(self):
        c = self.c; tok = self.peek(); r = dict(self.readings(tok))
        if c.wh[1] == "count" and "aux_be" in r and self.peek(1) == "there":        # how many N are there ?
            c.past = r["aux_be"] == "past"; self.i += 2; c.there = True
            c.subj, c.subj_cls = c.wh[0], c.wh_cls; c.wh = ("いくつ", "count")
            self.tail_pp_adv(); return self.end()
        if c.wh[1] in ("o", "count") and c.wh[0] not in ("何", "いくつ", "いくら") and not ("aux_do" in r or "modal" in r or "aux_have" in r) and ("verb" in r or "aux_be" in r and not (self.peek(1) and self.starts_np(self.peek(1)))):   # which bus goes there ?
            c.subj, c.subj_cls = c.wh[0], "t"; c.wh = (c.wh[0], "s")
            self.phase = "VP0"; return self.predicate()
        if c.wh[1] == "s" and ("verb" in r and "aux_be" not in r or "aux_be" in r and not (self.peek(1) and self.starts_np(self.peek(1)))):              # who came ? / who is in the room ?
            c.subj, c.subj_cls = None, "h"
            self.phase = "VP0"; return self.predicate()
        if ("aux_be" in r or "aux_be3" in r) and self.peek(1) and (self.starts_np(self.peek(1)) or self.has(self.peek(1), "wh")):
            tense = r.get("aux_be", "pres"); self.i += 1; c.past = tense == "past"
            ja, cls = self.parse_np(); c.subj, c.subj_cls = ja, cls
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            while self.peek() and self.has(self.peek(), "time"): c.time += dict(self.readings(self.peek()))["time"]; self.i += 1
            self.copula_predicate(allow_empty=True)
            if c.copula is None: c.copula = ("wh",)
            return self.end()
        if "aux_do" in r or "modal" in r or "aux_have" in r or "aux_be" in r:
            return self.question_inversion()
        self.fail("wh question")

    def question_inversion(self):
        c = self.c; tok = self.peek(); r = dict(self.readings(tok))
        c.question = True
        if "aux_do" in r:
            c.past = r["aux_do"] == "past"; self.i += 1
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            ja, cls = self.parse_np(); c.subj, c.subj_cls = ja, cls
            return self.verb_phrase(expect_base=True)
        if "modal" in r:
            c.modal, _ = r["modal"]; self.i += 1
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            ja, cls = self.parse_np(); c.subj, c.subj_cls = ja, cls
            while self.peek() and self.has(self.peek(), "adv") and not self.has(self.peek(), "verb") and not self.has(self.peek(), "aux_be"):
                c.adv += dict(self.readings(self.peek()))["adv"]; self.i += 1
            if tok == "would" and self.peek() == "mind" and self.peek(1) and self.verb_form(self.peek(1)) == "ing":   # would you mind V-ing ?
                self.i += 1; c.want = "mind_request"; c.modal = None
                return self.verb_phrase(expect_ing=True)
            return self.verb_phrase(expect_base=True)
        if "aux_have" in r:
            c.past = r["aux_have"] == "past"; c.perf = True; self.i += 1
            ja, cls = self.parse_np(); c.subj, c.subj_cls = ja, cls
            return self.verb_phrase(expect_pp=True)
        if "aux_be" in r:
            c.past = r["aux_be"] == "past"; self.i += 1
            ja, cls = self.parse_np(); c.subj, c.subj_cls = ja, cls
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            self.copula_predicate()
            return self.end()
        self.fail("inversion")

    def predicate(self):
        """after the subject"""
        c = self.c; tok = self.peek(); r = dict(self.readings(tok))
        while "adv" in r and "verb" not in r and "aux_be" not in r:
            c.adv += r["adv"]; self.i += 1; tok = self.peek(); r = dict(self.readings(tok))
        if "time" in r: c.time += r["time"]; self.i += 1; tok = self.peek(); r = dict(self.readings(tok))
        if "aux_be" in r or "aux_be3" in r:
            c.past = (r.get("aux_be") == "past"); self.i += 1
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            self.copula_predicate()
            return self.end()
        if "aux_do" in r and self.peek(1) and self.has(self.peek(1), "not"):
            c.past = r["aux_do"] == "past"; self.i += 2; c.neg = True
            return self.verb_phrase(expect_base=True)
        if "aux_do" in r and self.peek(1) and self.verb_form(self.peek(1)) == "base" and not self.has(self.peek(1), "noun"):   # emphatic: did come
            c.past = r["aux_do"] == "past"; self.i += 1; c.adv += "確かに"
            return self.verb_phrase(expect_base=True)
        if tok == "had" and self.peek(1) == "better":                    # had better V
            self.i += 2; c.want = "better"
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            return self.verb_phrase(expect_base=True)
        if tok == "need" and self.peek(1) and self.has(self.peek(1), "not"):   # need not V
            self.i += 2; c.want = "neednot"
            return self.verb_phrase(expect_base=True)
        if "modal" in r:
            c.modal, c.past = r["modal"]; self.i += 1
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            if self.peek() == "to": self.i += 1
            while self.peek() and self.has(self.peek(), "adv") and not self.has(self.peek(), "verb") and not self.has(self.peek(), "aux_be"):
                c.adv += dict(self.readings(self.peek()))["adv"]; self.i += 1
            if c.modal == "would" and self.peek() == "mind" and self.peek(1) and self.verb_form(self.peek(1)) == "ing":   # would you mind V-ing ?
                self.i += 1; c.want = "mind_request"; c.modal = None
                return self.verb_phrase(expect_ing=True)
            if self.peek() in ("have",) and self.peek(1) and self.verb_form(self.peek(1)) == "pp":
                self.i += 1; c.past = True; c.perf = True
                return self.verb_phrase(expect_pp=True)
            if self.has(self.peek(), "aux_be") and self.peek(1) and not self.has(self.peek(1), "verb"):   # will be late
                self.i += 1
                self.copula_predicate(); return self.end()
            return self.verb_phrase(expect_base=True)
        if ("aux_have" in r or tok == "'ve") and self.peek(1) and (self.has(self.peek(1), "not") and self.peek(2) and self.has(self.peek(2), "verb") or "pp" in self.verb_forms(self.peek(1))) and self.peek(1) != "to":
            c.past = True; c.perf = True; self.i += 1
            if self.has(self.peek(), "not"): self.i += 1; c.neg = True
            return self.verb_phrase(expect_pp=True)
        if ("aux_have" in r or tok == "'ve") and self.peek(1) and self.has(self.peek(1), "adv") and not self.has(self.peek(1), "verb"):   # have just now finished
            j = 1
            while self.peek(j) and self.has(self.peek(j), "adv") and not self.has(self.peek(j), "verb"): j += 1
            if "pp" in self.verb_forms(self.peek(j)):
                c.past = True; c.perf = True; self.i += 1
                while self.peek() and self.has(self.peek(), "adv") and not self.has(self.peek(), "verb"):
                    c.adv += dict(self.readings(self.peek()))["adv"]; self.i += 1
                return self.verb_phrase(expect_pp=True)
        if tok in ("have", "has", "had") and self.peek(1) == "to":          # have to
            c.past = tok == "had"; c.modal = "must"; self.i += 2
            return self.verb_phrase(expect_base=True)
        if "verb" in r:
            return self.verb_phrase()
        self.fail("predicate")

    def verb_forms(self, tok):
        return {f for cc, (l, f, ja, k) in [(cc, p) for cc, p in self.readings(tok) if cc == "verb"]}
    def verb_form(self, tok):
        fs = [f for cc, (l, f, ja, k) in [(cc, p) for cc, p in self.readings(tok) if cc == "verb"]]
        return fs[0] if fs else None

    def copula_predicate(self, allow_empty=False):
        c = self.c; tok = self.peek(); r = dict(self.readings(tok))
        while "adv" in r and (r["adv"] in ("とても", "本当に", "かなり", "も", "まだ", "もう", "いつも") or self.peek(1) and self.has(self.peek(1), "adj") and not self.has(self.peek(1), "noun")):
            c.degree = "すぎ" if tok == "too" else c.degree + r["adv"]; self.i += 1; tok = self.peek(); r = dict(self.readings(tok))
        if tok == "going" and self.peek(1) == "to" and self.peek(2) and self.has(self.peek(2), "verb"):
            self.i += 2; c.want = "intend"
            return self.verb_phrase(expect_base=True, from_copula=True)
        if tok == "to" and self.peek(1) and self.verb_form(self.peek(1)) == "base":        # he is to arrive tomorrow
            self.i += 1; c.want = "be_to"
            return self.verb_phrase(expect_base=True, from_copula=True)
        if tok == "as" and self.peek(1) and self.has(self.peek(1), "adj") and self.peek(2) == "as" and self.peek(3) and self.starts_np(self.peek(3)):
            adj = dict(self.readings(self.peek(1)))["adj"]; self.i += 3
            ja, cls = self.parse_np(); c.compare = ("as", ja); c.copula = ("adj", adj); self.tail_pp_adv(); return
        if tok == "more" and self.peek(1) and self.has(self.peek(1), "adj"):
            ja, k = dict(self.readings(self.peek(1)))["adj"]; self.i += 1
            self.toks[self.i] = self.toks[self.i]        # keep token; reading below sees the plain adjective
            c.degree += "より"
        if "verb" in r and self.verb_form(tok) == "ing":                     # progressive
            c.prog = True
            return self.verb_phrase(expect_ing=True, from_copula=True)
        if (tok, self.peek(1)) in ADJ_PREP and self.peek(2) and (self.starts_np(self.peek(2)) or self.verb_form(self.peek(2)) == "ing"):
            ja, kind, part = ADJ_PREP[(tok, self.peek(1))]; self.i += 2
            ja2, cls2 = self.parse_np(); c.obj = ja2; c.obj_particle = part
            c.copula = ("idiom", ja, kind); self.tail_pp_adv(); return
        if "verb" in r and self.verb_form(tok) == "pp" and not self.starts_np(tok):   # passive
            self.fail("passive")
        if (tok, self.peek(1)) in ADJ_PREP and self.peek(2) and (self.starts_np(self.peek(2)) or self.verb_form(self.peek(2)) == "ing"):
            ja, kind, part = ADJ_PREP[(tok, self.peek(1))]; self.i += 2
            if self.verb_form(self.peek()) == "ing":
                inner = Clause(); saved = self.c; self.c = inner
                self.verb_phrase(expect_ing=True, from_copula=True); self.c = saved
                c.obj = generate(inner, polite=False, embedded=True)[:-1] + "の"
            else:
                ja2, cls2 = self.parse_np(); c.obj = ja2
            c.obj_particle = part; c.copula = ("idiom", ja, kind); self.tail_pp_adv(); return
        if "adj" in r and not (self.peek(1) and "noun" in dict(self.readings(self.peek(1)))):
            c.copula = ("adj", r["adj"]); self.i += 1
            if self.peek() == "than" and self.peek(1) and (self.starts_np(self.peek(1)) or self.has(self.peek(1), "time")):
                self.i += 1
                if self.has(self.peek(), "time"): ja = dict(self.readings(self.peek()))["time"]; self.i += 1
                else: ja, cls = self.parse_np()
                c.compare = ("than", ja); self.tail_pp_adv(); return
            if self.peek() == "enough" and self.peek(1) == "to" and self.peek(2) and self.has(self.peek(2), "verb"):
                self.i += 2; inner = Clause(); saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                c.infinitive = ("enough", inner); self.tail_pp_adv(); return
            if c.degree == "すぎ" and self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb"):
                self.i += 1; inner = Clause(); saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                c.infinitive = ("too", inner); self.tail_pp_adv(); return
            if self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb"):   # glad to see / hard to say
                self.i += 1; inner = Clause(); saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                c.infinitive = ("extraposed" if c.subj in ("それ", None) else "adj_to", inner); self.tail_pp_adv(); return
            if self.peek() == "for" and self.peek(1) and self.starts_np(self.peek(1)):      # it is hard for X to V
                save_i = self.i; self.i += 1
                try:
                    xja, xcls = self.parse_np()
                except ParseFail:
                    self.i = save_i; xja = None
                if xja is not None and self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb"):
                    self.i += 1; inner = Clause(); inner.subj = xja; inner.subj_cls = xcls
                    saved = self.c; self.c = inner
                    self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                    c.infinitive = ("extraposed", inner); self.tail_pp_adv(); return
                self.i = save_i
            if self.peek() == "that" or (self.peek() and self.starts_np(self.peek()) and self.peek(1) and (self.has(self.peek(1), "verb") or self.has(self.peek(1), "aux_be") or self.has(self.peek(1), "modal") or self.has(self.peek(1), "aux_do"))):
                if self.peek() == "that": self.i += 1
                inner = Clause(); self.c = inner
                ja, cls = self.parse_np(); inner.subj, inner.subj_cls = ja, cls
                self.phase = "VP0"; self.predicate(); self.c = c
                if c.subj == "それ": c.infinitive = ("extraposed", inner)       # it is true that S -> Sのは本当だ
                else: c.embedded = inner
                return
            self.tail_pp_adv(); return
        if "prep" in r:
            p = r["prep"]; self.i += 1
            self.np_role = "pp_time" if tok in ("before", "after", "at", "until", "till", "around", "by") else None
            ja, cls = self.parse_np(); self.np_role = None
            c.copula = ("pp", p, ja); self.tail_pp_adv(); return
        if "adv" in r:                                                       # is here / is out
            c.copula = ("adv", r["adv"]); self.i += 1; self.tail_pp_adv(); return
        if "end" in r and allow_empty:
            return
        if self.starts_np(tok):
            ja, cls = self.parse_np()
            c.copula = ("np", ja, cls)
            if c.subj == "それ" and self.peek() == "that" and self.clause_follows(1):          # it is a pity that S
                self.i += 1; inner = Clause(); self.c = inner
                ja2, cls2 = self.parse_np(); inner.subj, inner.subj_cls = ja2, cls2
                self.phase = "VP0"; self.predicate(); self.c = c
                c.infinitive = ("extraposed", inner); return
            self.tail_pp_adv(); return
        self.fail("copula predicate")

    def verb_phrase(self, expect_base=False, expect_pp=False, expect_ing=False, from_copula=False):
        c = self.c; tok = self.peek(); r = dict(self.readings(tok))
        if "verb" not in r: self.fail("verb")
        readings = [p for cc, p in self.readings(tok) if cc == "verb"]
        lemma, form, ja, conj = readings[0]
        if lemma == "be" and not from_copula:
            self.i += 1
            self.copula_predicate(); return self.end()
        if lemma == "be" and from_copula:
            self.i += 1
            self.copula_predicate(); return
        self.i += 1
        if not expect_base and not expect_pp and not expect_ing:
            c.past = form == "past"
        c.verb = (lemma, ja, conj)
        for k in (3, 2, 1):                       # multiword verb idioms: "get in touch with", "look forward to", "wait for"
            key = (lemma, tuple(self.toks[self.i:self.i + k]))
            if key in VERB_IDIOM and self.i + k < len(self.toks) and (self.starts_np(self.toks[self.i + k]) or self.verb_form(self.toks[self.i + k]) == "ing"):
                ja, conj, part = VERB_IDIOM[key]; self.i += k
                c.verb = (lemma + " " + " ".join(key[1]), ja, conj); c.obj_particle = part
                if self.verb_form(self.peek()) == "ing":          # look forward to seeing you
                    inner = Clause(); saved = self.c; self.c = inner
                    self.verb_phrase(expect_ing=True, from_copula=True); self.c = saved
                    c.obj = generate(inner, polite=False, embedded=True)[:-1] + "の"
                else:
                    ja2, cls2 = self.parse_np(); c.obj = ja2
                self.tail_pp_adv()
                if from_copula: return
                return self.end()
        if (lemma, self.peek()) in PHRASAL and not (self.peek() in PREPS and self.peek(1) and self.starts_np(self.peek(1)) and self.peek() in ("for", "at", "to", "from", "with", "on") and (lemma, self.peek()) not in (("look", "for"), ("look", "at"), ("wait", "for"), ("listen", "to"), ("get", "on"), ("get", "to"), ("look", "after"), ("agree", "with"), ("depend", "on"), ("belong", "to"), ("hear", "from"), ("call", "on"))):
            ja, conj = PHRASAL[(lemma, self.peek())]; self.i += 1
            c.verb = (lemma + " " + self.toks[self.i - 1], ja, conj)
        if lemma == "go" and self.peek() in USE and "go" in USE[self.peek()]:
            uja, uconj, upart, _ = USE[self.peek()][lemma]
            c.obj = LEX[self.peek()][0][1]; c.obj_particle = upart; self.i += 1
            c.verb = ("go+" + self.toks[self.i - 1], uja, uconj)
            self.tail_pp_adv(); return self.end()
        if lemma == "let" and self.peek() in ("me", "us") and self.peek(1) and self.has(self.peek(1), "verb"):
            self.i += 1; c.imp = True; c.please = True; c.causative = True
            return self.verb_phrase(expect_base=True)
        # complements that change the predicate
        if lemma == "want" and self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb"):
            self.i += 1; c.want = "tai"
            return self.verb_phrase(expect_base=True)
        if lemma == "would like" or (lemma == "like" and c.modal == "will" and self.peek() == "to"):
            self.i += 1; c.want = "tai"; c.modal = None
            return self.verb_phrase(expect_base=True)
        if lemma in ("like", "love", "enjoy", "hate", "start", "begin", "stop", "finish", "try", "keep", "forget", "remember", "decide", "hope", "intend", "need", "plan", "learn", "promise", "fail", "continue", "help", "expect", "seem", "ought", "refuse", "agree", "manage", "pretend", "wish", "mean", "choose", "prefer", "happen", "tend", "afford", "offer", "dare") and self.peek() in ("to",) and self.peek(1) and self.has(self.peek(1), "verb"):
            self.i += 1; c.want = lemma
            return self.verb_phrase(expect_base=True)
        if lemma in ("promise", "try", "decide", "tell", "ask", "want", "prefer", "choose", "pretend", "hope", "manage") and self.has(self.peek(), "not") and self.peek(1) == "to" and self.peek(2) and self.has(self.peek(2), "verb"):   # promised not to tell
            self.i += 2; c.want = lemma; c.neg_inner = True
            return self.verb_phrase(expect_base=True)
        if lemma in ("like", "love", "enjoy", "hate", "start", "begin", "stop", "finish", "keep", "practice", "mind") and self.peek() and self.verb_form(self.peek()) == "ing":
            c.want = lemma
            return self.verb_phrase(expect_ing=True)
        if lemma in ("think", "say", "know", "believe", "hope", "hear", "tell", "feel", "suppose", "guess", "wish", "expect", "remember", "forget") and (self.peek() == "that" or (self.peek() and self.starts_np(self.peek()) and lemma not in ("tell", "know", "hear", "remember", "forget", "feel") and self.peek(1) and (self.has(self.peek(1), "verb") or self.has(self.peek(1), "aux_be") or self.has(self.peek(1), "aux_do") or self.has(self.peek(1), "modal") or self.has(self.peek(1), "aux_have")))):
            if self.peek() == "that": self.i += 1
            inner = Clause(); c.embedded = inner
            self.c = inner
            ja, cls = self.parse_np(); inner.subj, inner.subj_cls = ja, cls
            self.phase = "VP0"
            self.predicate_noend()
            self.c = c
            return self.end()
        if lemma == "tell" and self.peek() and self.starts_np(self.peek()) and not self.next_head_uses(lemma):
            ja, cls = self.parse_np(); c.iobj = ja
            if self.peek() == "that": self.i += 1; return self.embedded_after(c)
            if self.peek() and (self.has(self.peek(), "wh") or self.peek() in ("if", "whether")):
                c.obj = self.embedded_question(); c.obj_particle = ""; self.tail_pp_adv()
                return self.end() if not from_copula else None
        if lemma in ("know", "wonder", "ask", "see", "understand", "show", "remember", "forget", "explain", "learn", "decide", "imagine", "care", "find out", "tell", "say", "guess", "think") and self.peek() and (self.has(self.peek(), "wh") or self.peek() in ("if", "whether")) and not (self.peek() == "if" and lemma not in ("wonder", "ask", "know", "see", "care")):
            if lemma == "ask" and self.peek() and self.starts_np(self.peek()): ja, cls = self.parse_np(); c.iobj = ja
            c.obj = self.embedded_question(); c.obj_particle = ""
            if lemma == "wonder": c.verb = ("wonder", "と思う", "fixed"); c.obj_particle = ""
            self.tail_pp_adv()
            return self.end() if not from_copula else None
        if lemma == "ask" and self.peek() and self.starts_np(self.peek()) and self.peek(1) and (self.has(self.peek(1), "wh") or self.peek(1) in ("if", "whether")) and not self.next_head_uses(lemma):
            ja, cls = self.parse_np(); c.iobj = ja
            c.obj = self.embedded_question(); c.obj_particle = ""; self.tail_pp_adv()
            return self.end() if not from_copula else None
            if self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb"):
                self.fail("tell someone to")
        if lemma == "give" and self.peek() and self.starts_np(self.peek()) and self.peek(1) and self.starts_np(self.peek(1)):
            ja, cls = self.parse_np(); c.iobj = ja
        self.tail()
        if from_copula:
            return
        return self.end()

    def sub_clause(self, conj):
        """'conj S V ...' -> Japanese subordinate clause string, or None (input restored) if it does not parse"""
        save = (self.i, self.c, self.phase, self.np_role)
        self.i += 1
        inner = Clause(); inner.sub = True; self.c = inner
        try:
            if self.verb_form(self.peek()) == "ing" and conj in ("while", "before", "after", "when", "since", "once"):     # while walking
                inner.subj = None
                self.verb_phrase(expect_ing=True, from_copula=True)
            elif self.peek() == "there" and self.peek(1) and self.has(self.peek(1), "aux_be"):
                self.i += 1; r = dict(self.readings(self.peek())); inner.past = r["aux_be"] == "past"; self.i += 1; inner.there = True
                ja, cls = self.parse_np(); inner.subj, inner.subj_cls = ja, cls; self.tail_pp_adv()
            else:
                ja, cls = self.parse_np(); inner.subj, inner.subj_cls = ja, cls
                self.phase = "VP0"; self.predicate()
        except ParseFail:
            self.i, self.c, self.phase, self.np_role = save
            return None
        self.c, self.np_role = save[1], save[3]
        if self.phase != "DONE": self.phase = save[2]
        if self.peek() == ",": self.i += 1
        if self.peek() == "then": self.i += 1
        return subord_form(inner, conj)

    def coord(self, conj):
        """main clause + conj + clause: the first clause becomes a te-form (and) / が (but) / ので (so) link"""
        c = self.c
        save = (self.i, self.phase)
        self.i += 1
        second = Clause(); second.coord = True
        self.c = second
        try:
            if self.peek() and self.starts_np(self.peek()) and self.peek(1) and (self.has(self.peek(1), "verb") or self.has(self.peek(1), "aux_be") or self.has(self.peek(1), "aux_do") or self.has(self.peek(1), "aux_have") or self.has(self.peek(1), "modal")):
                ja, cls = self.parse_np(); second.subj, second.subj_cls = ja, cls
            else:
                second.subj = None; second.subj_cls = c.subj_cls
            if self.has(self.peek(), "not") and self.peek(1) and self.has(self.peek(1), "verb"): self.i += 1; second.neg = True
            self.phase = "VP0"; self.predicate()
        except ParseFail:
            self.c = c; self.i, self.phase = save
            return False
        self.c = c
        if self.phase != "DONE": self.phase = save[1]
        first = Clause(); first.__dict__.update(c.__dict__); first.pre_clause = ""; first.suffix = ""
        link = COORD[conj]
        if link == "":
            head = clause_te(first)
        else:
            head = generate(first, polite=False)[:-1] + link
        pre = c.pre_clause
        c.__dict__.update(second.__dict__)
        c.imp = first.imp or second.imp; c.please = first.please or second.please; c.lets = first.lets
        c.pre_clause = pre + head
        return True

    def embedded_question(self):
        """where the bus stop is / which you want / if he will come -> ...か / ...かどうか"""
        tok = self.peek(); r = dict(self.readings(tok))
        inner = Clause(); inner.sub = True; saved = self.c; self.c = inner
        if tok in ("if", "whether"):
            self.i += 1; ja, cls = self.parse_np(); inner.subj, inner.subj_cls = ja, cls
            self.phase = "VP0"; self.predicate(); self.c = saved
            body = generate(inner, polite=False, embedded=True)[:-1]
            return body + "かどうか"
        inner.wh = r["wh"]; inner.question = True; self.i += 1
        nxt = self.peek(); nr = dict(self.readings(nxt)) if nxt else {}
        if tok in ("what", "which") and ("noun" in nr or "adj" in nr) and not ("verb" in nr or "aux_be" in nr):
            ja, cls = self.parse_np(); inner.wh = (("どんな" if tok == "what" else "どの") + ja, "o")
        elif tok == "how" and nxt in ("many", "much") and self.peek(1) and self.starts_np(self.peek(1)):
            self.i += 1; ja, cls = self.parse_np(); inner.wh = ("いくつの" + ja, "o")
        elif tok == "how" and ("adj" in nr or "adv" in nr):
            inner.wh = ("何歳" if nxt == "old" else "どのくらい", "adv"); self.i += 1
        if inner.wh[1] in ("o", "loc", "adv", "t") and self.peek() == "to" and self.peek(1) and self.verb_form(self.peek(1)) == "base":   # which way to go / how to do it
            self.i += 1; self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
            body = generate(inner, polite=False, embedded=True)[:-1]
            return (body[:-1] if body.endswith("か") else body) + "べきか"
        if self.peek() and self.starts_np(self.peek()):
            ja, cls = self.parse_np(); inner.subj, inner.subj_cls = ja, cls
            nxt = self.peek(); nr = dict(self.readings(nxt)) if nxt else {}
            after = self.peek(1); ar = dict(self.readings(after)) if after else {}
            if "aux_be" in nr and (after is None or "end" in ar or "conj" in ar or after == ","):    # where the bus stop is
                inner.past = nr["aux_be"] == "past"; self.i += 1; inner.copula = ("wh",)
            else:
                self.phase = "VP0"; self.predicate()
        else:                                                   # what had happened / who came
            inner.subj = None; inner.wh = (inner.wh[0], "s"); self.phase = "VP0"; self.predicate()
        self.c = saved
        body = generate(inner, polite=False, embedded=True)[:-1]
        return body if body.endswith("か") else body + "か"

    def embedded_after(self, c):
        inner = Clause(); c.embedded = inner; self.c = inner
        ja, cls = self.parse_np(); inner.subj, inner.subj_cls = ja, cls
        self.phase = "VP0"; self.predicate_noend(); self.c = c
        return self.end()

    def predicate_noend(self):
        """predicate of an embedded clause; end() returns without consuming for an inner clause"""
        self.predicate()

    def tail(self):
        """object, indirect object, particles, PPs, adverbs after a verb"""
        c = self.c
        tok = self.peek(); r = dict(self.readings(tok))
        # came to help me / went to see her: purpose infinitive after a motion verb
        if tok == "to" and self.peek(1) and self.has(self.peek(1), "verb") and self.verb_form(self.peek(1)) == "base" and c.verb and c.verb[0] in ("come", "go", "return", "stop", "stay", "leave", "run", "walk", "hurry"):
            self.i += 1
            inner = Clause(); inner.purpose = True
            saved = self.c; self.c = inner
            self.verb_phrase(expect_base=True, from_copula=True)
            self.c = saved; c.purpose_clause = inner
            self.tail_pp_adv(); return
        if tok == "to" and self.peek(1) and self.has(self.peek(1), "verb") and self.verb_form(self.peek(1)) == "base" and c.verb:
            self.i += 1
            inner = Clause(); inner.purpose = "tame"
            saved = self.c; self.c = inner
            self.verb_phrase(expect_base=True, from_copula=True)
            self.c = saved; c.purpose_clause = inner
            self.tail_pp_adv(); return
        if tok == "to" and self.peek(1) and self.starts_np(self.peek(1)) and c.verb and c.verb[0] in ("give", "send", "show", "tell", "lend", "bring", "write", "say", "teach", "explain", "pay"):
            pass
        if tok == "both" and self.peek(1) and self.starts_np(self.peek(1)):           # both A and B
            self.i += 1; a, _ = self.parse_np()
            if self.peek() == "and": self.i += 1; b, _ = self.parse_np(); c.obj = a + "も" + b; c.obj_particle = "も"
            else: c.obj = "両方の" + a
            self.tail_pp_adv(); return
        if self.verb_form(tok) == "ing" and c.verb and c.verb[0] in ("stand", "sit", "lie", "come", "go", "spend", "walk", "run", "wait"):   # stood looking at
            inner = Clause(); saved = self.c; self.c = inner
            self.verb_phrase(expect_ing=True, from_copula=True); self.c = saved
            c.manner = inner; self.tail_pp_adv(); return
        if self.starts_np(tok) and "prep" not in r and ("adv" not in r or self.peek(1) and self.has(self.peek(1), "noun") and not (self.peek(2) and self.starts_np(self.peek(2)))) and not ("adj" in r and "noun" not in r and "det" not in r and c.verb and c.verb[0] in ("get", "become", "feel", "look", "seem", "grow", "turn")):
            ja, cls = self.parse_np(); c.obj = ja
            head = self.last_head
            lemma0 = c.verb[0] if c.verb else None
            if c.verb and c.verb[0] in ("make", "let", "have") and self.peek() and self.verb_form(self.peek()) == "base" and not self.has(self.peek(), "noun"):   # made me write
                inner = Clause(); inner.subj = ja; inner.subj_cls = cls
                saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True); self.c = saved
                c.obj = None; c.control = inner; c.causative = c.verb[0]
                self.tail_pp_adv(); return
            if c.verb and c.verb[0] in ("see", "hear", "watch", "feel", "find", "notice", "smell") and self.peek() and self.verb_form(self.peek()) == "ing":   # hear someone talking
                inner = Clause(); inner.subj = ja; inner.subj_cls = cls
                saved = self.c; self.c = inner
                self.verb_phrase(expect_ing=True, from_copula=True); self.c = saved
                c.obj = generate(inner, polite=False, embedded=True)[:-1] + "の"
                self.tail_pp_adv(); return
            if head in USE and lemma0 in USE[head] and not c.want:        # the noun's use-verb replaces the light verb
                uja, uconj, upart, stative = USE[head][lemma0]
                c.verb = (lemma0 + "+" + head, uja, uconj); c.obj_particle = upart
                if upart == "-": c.obj = None; c.obj_particle = None           # take a rest -> 休む: the noun is the verb
                if stative: c.prog = True
            # told me to go / want you to come / asked him to wait
            if self.peek() == "to" and self.peek(1) and self.has(self.peek(1), "verb") and self.verb_form(self.peek(1)) == "base" and c.verb and c.verb[0] in ("tell", "ask", "want", "advise", "allow", "expect", "order", "help", "teach", "get", "force", "like"):
                self.i += 1
                inner = Clause(); inner.subj = c.obj; inner.subj_cls = cls
                saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True)
                self.c = saved; c.obj = None; c.control = inner
                self.tail_pp_adv(); return
        self.tail_pp_adv()

    def tail_pp_adv(self):
        c = self.c
        while True:
            tok = self.peek(); r = dict(self.readings(tok))
            if tok is None or "end" in r: return
            if c.sub and (tok == "," or "conj" in r and not (tok in ("as", "than", "until", "since", "after", "before") and self.peek(1) and self.starts_np(self.peek(1)) and not self.clause_follows(1))): return
            if tok == "," and self.peek(1) and self.has(self.peek(1), "pron") and self.peek(2) in HEARSAY and self.peek(3) and self.has(self.peek(3), "end"):   # ..., i hear .
                c.suffix = HEARSAY[self.peek(2)]; self.i += 3; continue
            if "conj" in r and tok in COORD and c is self.main:
                if tok == "while" and self.clause_follows(1):
                    s = self.sub_clause(tok)
                    if s is not None: c.pre_clause += s; continue
                if self.coord(tok): continue
            if "conj" in r and tok in SUBORD and c is self.main and (tok not in PREPS or self.clause_follows(1)):
                s = self.sub_clause(tok)
                if s is not None: c.pre_clause += s; continue
            if tok == "," and self.peek(1) and self.has(self.peek(1), "conj"): self.i += 1; continue
            if "time" in r: c.time += r["time"]; self.i += 1; continue
            if "prep" in r and self.peek(1) and self.has(self.peek(1), "time") and not self.starts_np(self.peek(1)):
                p = r["prep"]; self.i += 1; tja = dict(self.readings(self.peek()))["time"]; self.i += 1
                p = {"until": ("まで", "まで"), "till": ("まで", "まで"), "by": ("までに", "までに"), "since": ("から", "から"), "from": ("から", "から"), "before": ("の前に", "の前に"), "after": ("の後で", "の後で")}.get(tok, p)
                c.pp.append((p, tja)); continue
            if "prep" in r and self.peek(1) and self.starts_np(self.peek(1)):
                p = r["prep"]; self.i += 1
                self.np_role = "pp_time" if tok in ("before", "after", "at", "until", "till", "around", "by") else None
                ja, cls = self.parse_np(); self.np_role = None
                if cls == "t" and ja.endswith("時"):
                    p = {"at": ("に", "に"), "before": ("前に", "前に"), "after": ("後に", "後に"), "around": ("ごろに", "ごろに"), "by": ("までに", "までに")}.get(tok, p)
                if tok == "with" and cls == "t" and c.verb and c.verb[0] not in ("live", "stay", "play", "talk", "speak", "agree", "go", "come", "be", "eat", "work"):
                    p = ("で", "で")                                               # instrument: with a pen -> ペンで
                if p == PREPS["to"] and c.verb and c.verb[0] in ("give", "send", "show", "tell", "lend", "bring", "write", "say", "teach", "explain", "pay"):
                    c.iobj = ja
                else:
                    c.pp.append((p, ja))
                continue
            if tok in ("on", "off", "up", "down", "out", "away", "back", "in") and c.verb and (c.verb[0], tok) in PARTICLE_AFTER and (self.peek(1) is None or "end" in dict(self.readings(self.peek(1))) or "prep" in dict(self.readings(self.peek(1))) or "time" in dict(self.readings(self.peek(1)))):
                ja, conj = PARTICLE_AFTER[(c.verb[0], tok)]; c.verb = (c.verb[0] + " " + tok, ja, conj); self.i += 1; continue
            if "prep" in r and c.wh and self.peek(1) and "end" in dict(self.readings(self.peek(1))):   # where do you come from ?
                c.wh = (c.wh[0] + r["prep"][1 if c.wh[1] == "loc" else 0], "pp"); self.i += 1; continue
            if tok == "to" and self.peek(1) and self.has(self.peek(1), "verb") and self.verb_form(self.peek(1)) == "base" and c.verb and c.purpose_clause is None and c.control is None:
                self.i += 1                                      # purpose infinitive after a prepositional phrase
                inner = Clause(); inner.purpose = "tame" if c.verb[0] not in ("come", "go", "return", "stop", "stay", "leave", "run", "walk", "hurry") else True
                saved = self.c; self.c = inner
                self.verb_phrase(expect_base=True, from_copula=True)
                self.c = saved; c.purpose_clause = inner; continue
            if tok == "as" and self.peek(1) and self.has(self.peek(1), "adv") and self.peek(2) == "as" and self.peek(3) and self.starts_np(self.peek(3)):   # as early as my mother
                c.adv += dict(self.readings(self.peek(1)))["adv"]; self.i += 3
                ja, cls = self.parse_np(); c.compare = ("as", ja); continue
            if tok == "than" and self.peek(1) and (self.starts_np(self.peek(1)) or self.has(self.peek(1), "time")):
                self.i += 1
                if self.has(self.peek(), "time"): ja = dict(self.readings(self.peek()))["time"]; self.i += 1
                else: ja, cls = self.parse_np()
                c.compare = ("than", ja); continue
            if ("num" in r or tok in ("a", "an", "several", "many", "few", "some")) and self.peek(1) in DURATION:   # two years / several weeks / a few days
                n = r["num"] if "num" in r else {"a": "一", "an": "一", "several": "数", "many": "何", "few": "数", "some": "数"}[tok]
                c.adv += n + DURATION[self.peek(1)]; self.i += 2; continue
            if tok == "for" and self.peek(1) and ("num" in dict(self.readings(self.peek(1))) or self.peek(1) in ("a", "an", "several", "many", "few", "some")) and self.peek(2) in DURATION:
                self.i += 1; continue
            if tok == "," and self.peek(1) and (self.peek(1) in CONTRACTED or self.has(self.peek(1), "aux_do") or self.has(self.peek(1), "aux_be") or self.has(self.peek(1), "modal")) and self.toks[-1] == "?" and len(self.toks) - self.i <= 5:   # ..., isn't it ?
                c.tag = True; self.i = len(self.toks) - 1; continue
            if "adj" in r and self.peek(1) == "and" and self.peek(2) == tok and c.verb and c.verb[0] in ("get", "become", "grow"):   # longer and longer
                ja, k = r["adj"]; c.adv += "ますます"; c.copula = ("adj", (ja[2:] if ja.startswith("より") else ja, k)); self.i += 3; continue
            if "adv" in r: c.adv += r["adv"]; self.i += 1; continue
            if "adj" in r and c.verb and c.verb[0] in ("get", "become", "feel", "look", "seem", "grow", "turn"):
                c.copula = ("adj", r["adj"]); self.i += 1; continue
            if tok == "," : self.i += 1; continue
            self.fail("tail")

    def clause_follows(self, k):
        """does a clause (NP then verb/aux) start at offset k?"""
        j = self.i + k
        if j >= len(self.toks) or not self.starts_np(self.toks[j]): return False
        while j < len(self.toks) and self.toks[j] not in (".", "?", "!", ","):
            r = dict(self.readings(self.toks[j]))
            if any(x in r for x in ("verb", "aux_be", "aux_do", "aux_have", "modal")) and not ("noun" in r and j == self.i + k): return True
            if "prep" in r or "conj" in r: return False
            j += 1
        return False

    def end(self):
        if self.phase == "DONE" and self.i >= len(self.toks):   # an inner clause already consumed the end token
            return
        tok = self.peek(); r = dict(self.readings(tok))
        if "end" in r:
            self.i += 1
            if r["end"] == "?": self.main.question = True
            if self.i < len(self.toks): self.fail("two sentences")
            self.phase = "DONE"; return
        if self.c is not self.main:            # embedded / relative clause: let the outer clause finish
            return
        self.fail("end")

# ============================================================================
# 5. Generation: frame -> Japanese, a fixed template of table lookups
# ============================================================================
def predicate_form(c, polite):
    """the finite predicate of a clause from its registers"""
    if c.there:
        base = Pred("v1", "い") if c.subj_cls in ("h", "a") else Pred("v5", "ある")
        return base.finite(c.past, c.neg, polite)
    if c.copula is not None:
        kind, *rest = c.copula
        if kind == "wh":
            return ""
        if kind == "idiom":
            ja, k = rest
            p = {"na": Pred("na", ja), "a": Pred("ia", ja[:-1]), "fx": adj_pred(ja, "fx"), "v": make_pred(ja, "i" if ja.endswith(("れる", "える", "ける")) else "g")}[k]
            return p.finite(c.past, c.neg, polite)
        if kind == "adj":
            ja, k = rest[0]
            if c.compare is not None and ja.startswith("より"): ja = ja[2:]
            p = adj_pred(ja, k)
        elif kind == "np":
            p = Pred("na", rest[0])
            if c.relative and not c.neg and not c.past: return rest[0] + "である"
        elif kind == "pp":
            prep, np = rest
            p = Pred("v1", "い") if c.subj_cls in ("h", "a") else Pred("v5", "ある")
        elif kind == "adv":
            p = Pred("v1", "い") if c.subj_cls in ("h", "a") else Pred("v5", "ある")
        if c.modal == "must": return p_must(p, c, polite)
        return p.finite(c.past, c.neg, polite)
    if c.verb is None:
        return ""
    lemma, ja, conj = c.verb
    if conj == "fixed":
        p = Pred("fixed", ja)
    elif conj == "ia_fixed":
        p = Pred("ia", ja[:-1])
    else:
        p = make_pred(ja, conj)
    if c.want == "tai":          p = p.tai()
    elif c.want in ("like", "love", "enjoy"): p = Pred("na", p.dic + "のが好き")
    elif c.want in ("hate",):    p = Pred("na", p.dic + "のが嫌い")
    elif c.want == "promise" and getattr(c, "neg_inner", False): p = Pred("fixed", p.nai_stem() + "ないと約束する")
    elif c.want == "try" and getattr(c, "neg_inner", False):     p = Pred("fixed", p.nai_stem() + "ないようにする")
    elif c.want == "decide" and getattr(c, "neg_inner", False):  p = Pred("fixed", p.nai_stem() + "ないことにした")
    elif c.want in ("start", "begin"): p = Pred("v1", p.masu_stem() + "始め", dic=p.masu_stem() + "始める")
    elif c.want in ("stop", "finish"): p = Pred("v1", p.dic + "のをやめ", dic=p.dic + "のをやめる")
    elif c.want == "try":        p = Pred("v1", p.te() + "み", dic=p.te() + "みる")
    elif c.want == "keep":       p = Pred("v1", p.te() + "い", dic=p.te() + "いる")
    elif c.want == "forget":     p = Pred("v1", p.dic + "のを忘れ", dic=p.dic + "のを忘れる")
    elif c.want == "remember":   p = Pred("v1", p.dic + "のを覚え", dic=p.dic + "のを覚える")
    elif c.want == "decide":     p = Pred("fixed", p.dic + "ことにした")
    elif c.want == "hope":       p = Pred("fixed", p.dic + "ことを望む")
    elif c.want == "need":       p = Pred("fixed", p.dic + "必要がある")
    elif c.want == "plan":       p = Pred("na", p.dic + "つもり")
    elif c.want == "intend":     p = Pred("na", p.dic + "つもり")
    elif c.want == "promise":    p = Pred("fixed", p.dic + "と約束する")
    elif c.want == "learn":      p = Pred("v5", p.dic + "ことを学ぶ")
    elif c.want == "fail":       p = Pred("v1", p.dic + "のに失敗し", dic=p.dic + "のに失敗する")
    elif c.want == "continue":   p = Pred("v1", p.masu_stem() + "続け", dic=p.masu_stem() + "続ける")
    elif c.want == "help":       p = Pred("v5", p.dic + "のを手伝う")
    elif c.want == "expect":     p = Pred("fixed", p.dic + "と思う")
    elif c.want == "seem":       p = Pred("fixed", p.dic + "ようだ")
    elif c.want == "refuse":     p = Pred("fixed", _inner_dic(p, c) + "ことを拒否する")
    elif c.want == "agree":      p = Pred("fixed", _inner_dic(p, c) + "ことに同意する")
    elif c.want == "manage":     p = Pred("fixed", "何とか" + p.dic)
    elif c.want == "pretend":    p = Pred("fixed", _inner_dic(p, c) + "ふりをする")
    elif c.want == "wish":       p = p.tai()
    elif c.want == "mean":       p = Pred("na", _inner_dic(p, c) + "つもり")
    elif c.want == "choose":     p = Pred("fixed", _inner_dic(p, c) + "ことにする")
    elif c.want == "prefer":     p = Pred("na", _inner_dic(p, c) + "ほうが好き")
    elif c.want == "happen":     p = Pred("fixed", "偶然" + p.dic)
    elif c.want == "tend":       p = Pred("na", p.masu_stem() + "がち")
    elif c.want == "afford":     p = Pred("fixed", p.dic + "余裕がある")
    elif c.want == "offer":      p = Pred("fixed", p.volitional(False) + "と申し出る")
    elif c.want == "dare":       p = Pred("fixed", "あえて" + p.dic)
    elif c.want == "practice":   p = Pred("v1", p.dic + "練習をし", dic=p.dic + "練習をする")
    elif c.want == "mind":       p = Pred("fixed", p.dic + "のを気にする")
    elif c.want == "better":     p = Pred("ia", (p.nai_stem() + "ない" if c.neg else p.ta()) + "ほうがい"); c.neg = False
    elif c.want == "neednot":    return p.dic + "必要はない" if not polite else p.dic + "必要はありません"
    elif c.want == "mind_request": return p.te() + "もらえません"
    elif c.want == "be_to":      p = Pred("fixed", p.dic + "ことになっている")
    if "+" in lemma:
        lemma = "__use__"
    if lemma in STATIVE and not c.want and not c.imp and not c.lets and c.modal not in ("can",) and not c.perf:
        c.prog = True
    if lemma == "like" and not c.want:  p = Pred("na", "好き")
    elif lemma == "want" and not c.want and c.obj: p = Pred("ia", "欲し")
    elif lemma == "need" and not c.want: p = Pred("na", "必要")
    elif lemma == "hate" and not c.want: p = Pred("na", "嫌い")
    elif lemma == "understand" and not c.want: p = Pred("v5", "分かる")
    if c.perf and c.neg: c.prog = True                         # haven't come yet -> まだ来ていない
    if c.prog and not c.want and p.kind in ("v1", "v5", "suru", "kuru"):     p = p.progressive()
    if c.modal == "can":          p = p.potential() if p.kind in ("v1", "v5", "suru", "kuru") else p
    elif c.modal == "must":       return p_must(p, c, polite)
    elif c.modal == "should":     p = p.should()
    elif c.modal == "may":        p = p.maybe()
    if c.lets:                    return p.volitional(polite=True) if p.kind in ("v1", "v5", "suru", "kuru") else p.finite(False, False, True)
    if c.imp:
        if c.causative and p.kind in ("v1", "v5", "suru", "kuru"):
            caus = {"v1": p.stem + "させ", "v5": p.nai_stem() + "せ", "suru": p.stem[:-2] + "させ", "kuru": p.stem[:-1] + "させ"}[p.kind]
            return caus + "てください"
        if c.neg: return p.nai_stem() + "ないでください"
        return p.te() + "ください"
    if c.please and c.verb and not c.imp:
        return p.te() + "ください"
    past = c.past and not (c.modal == "will")
    return p.finite(past, c.neg, polite)

def _inner_dic(p, c):
    """dictionary form of the complement verb, negated when the English had 'not to V'"""
    return (p.nai_stem() + "ない") if getattr(c, "neg_inner", False) else p.dic

def _copy_for_hoshii(c):
    """want someone to V: the main predicate becomes ～てほしい (an i-adjective)"""
    c.verb = ("want someone to", "ほしい", "ia_fixed")
    return c

def adj_pred(ja, k):
    if k == "fx":
        return Pred("v1", ja[:-1]) if ja.endswith("いる") else Pred("fixed", ja)
    if ja.endswith("い") and not ja.endswith(("きれい", "ない")) and k != "na" or (k == "a" and ja.endswith("い") and not ja.endswith("ない")):
        return Pred("ia", ja[:-1])
    if ja.endswith("な") and k in ("a", "na"):
        return Pred("na", ja[:-1])
    if ja.endswith("の") or ja.endswith("た") or ja.endswith("て"):
        return Pred("fixed", ja[:-1] + "だ" if ja.endswith("の") else ja)
    return Pred("na", ja)

STATIVE = {"have", "know", "live", "wear", "love", "hold", "work"}
GA_VERBS = {"like", "want", "need", "hate", "understand"}
def p_must(p, c, polite):
    if p.kind in ("v1", "v5", "suru", "kuru"):
        if c.neg: return p.te() + "はいけない" + ("です" if polite else "")
        return p.nai_stem() + "なければならない" + ("です" if polite else "")
    return p.finite(False, False, False) + "はずだ"

def generate(c, polite=None, embedded=False):
    if polite is None:
        polite = bool(c.question)
    out = []
    if c.howabout is not None:
        return c.howabout + "はどうですか。"
    if c.exclaim is not None:
        return (c.subj + "は" if c.subj else "") + "なんと" + c.exclaim + "なのだろう。"
    if c.pre_clause: out.append(c.pre_clause)
    if c.time: out.append(c.time)
    subj_particle = "が" if (embedded and not c.coord) or (c.wh and c.wh[1] == "s") or c.there else "は"
    if c.there:
        for p, np in c.pp: out.append(np + p[1])
        if c.wh and c.wh[1] == "count":
            out.append(c.subj + "は" + "いくつ"); out.append(predicate_form(c, polite)); out.append("か")
            return "".join(out) + "。"
        out.append(c.subj + "が")
        out.append(predicate_form(c, polite))
    else:
        wh_cop = c.wh and c.copula is not None and c.copula[0] == "wh"
        if c.subj in ("誰も", "何も"): c.neg = True
        weather = c.subj == "それ" and (c.verb and c.verb[0] in ("rain", "snow") or c.infinitive and c.infinitive[0] == "extraposed")
        if (c.subj is not None or c.wh and c.wh[1] == "s") and not c.imp and not c.lets and not (wh_cop and c.wh[0] == "何時" and c.subj == "それ") and c.want != "mind_request" and not weather:
            if c.wh and c.wh[1] == "s" and not wh_cop: out.append(c.wh[0] + "が")
            elif c.subj in ("誰も", "何も"): out.append(c.subj)
            else: out.append(c.subj + ("は" if wh_cop and not embedded else subj_particle))
        if c.wh and c.wh[1] in ("t", "pp") and not wh_cop: out.append(c.wh[0] + ("に" if c.wh[0] == "何時" else ""))
        if c.wh and c.wh[1] == "loc" and not wh_cop: out.append("どこ" + ("に" if c.copula or c.verb and c.verb[0] in ("be", "live", "go", "come", "stay") else "で"))
        if c.wh and c.wh[1] == "adv" and not c.copula: out.append(c.wh[0] if c.wh[0] != "どう" else "どうやって")
        if c.wh and c.wh[1] == "count": out.append("いくつの" + c.wh[0] + "を")
        exist = c.copula is not None and c.copula[0] == "pp"
        for p, np in c.pp:
            out.append(np + p[1 if exist or (c.verb and c.verb[0] in ("live", "stay", "be", "go", "come", "arrive", "return", "get", "move", "sit", "stand", "put")) else 0])
        if c.copula is not None and c.copula[0] == "pp":
            out.append(c.copula[2] + c.copula[1][1])
        if c.compare is not None:
            kind_c, np_c = c.compare
            out.append(np_c + ("より" if kind_c == "than" else ("ほど" if c.neg else "と同じくらい")))
        if c.manner is not None:
            inner = c.manner
            body = generate(inner, polite=False, embedded=True)[:-1]
            ipred = predicate_form(inner, False)
            lemma0, ja0, conj0 = inner.verb
            p = make_pred(ja0, conj0) if conj0 != "fixed" else Pred("fixed", ja0)
            out.append(body[:-len(ipred)] + p.masu_stem() + "ながら")
        if c.control is not None and c.causative in ("make", "let", "have"):
            inner = c.control
            body = generate(inner, polite=False, embedded=True)[:-1]
            ipred = predicate_form(inner, False)
            lemma0, ja0, conj0 = inner.verb
            p = make_pred(ja0, conj0) if conj0 != "fixed" else Pred("fixed", ja0)
            out.append(inner.subj + ("に" if inner.obj else "を") + body[len(inner.subj) + 1:-len(ipred)] + p.causative().finite(c.past, c.neg, polite))
            c.verb = None; c.control = None
        if c.control is not None:
            inner = c.control
            want_to = c.verb[0] == "want"
            inner_pred = predicate_form(inner, False)
            lemma0, ja0, conj0 = inner.verb if inner.verb else (None, None, None)
            body = generate(inner, polite=False, embedded=True)[:-1]
            body = body[len(inner.subj) + 1:] if inner.subj else body       # drop the が-subject
            if want_to:
                p = make_pred(ja0, conj0) if conj0 != "fixed" else Pred("fixed", ja0)
                out.append(inner.subj + "に" + body[:-len(inner_pred)] + p.te() + "ほしい" if False else inner.subj + "に" + body[:-len(inner_pred)] + p.te())
                c = _copy_for_hoshii(c)
            else:
                out.append(inner.subj + "に" + body + "ように")
        if c.purpose_clause is not None:
            inner = c.purpose_clause
            body = generate(inner, polite=False, embedded=True)[:-1]
            inner_pred = predicate_form(inner, False)
            lemma0, ja0, conj0 = inner.verb
            p = make_pred(ja0, conj0) if conj0 != "fixed" else Pred("fixed", ja0)
            if inner.purpose == "tame":
                out.append(body + "ために")
            else:
                out.append(body[:-len(inner_pred)] + p.masu_stem() + "に")
        if c.iobj: out.append(c.iobj + "に")
        if c.obj and c.subj and c.subj in ("私", "彼", "彼女", "私たち", "彼ら", "あなた") and c.obj.startswith(c.subj + "の") and len(c.obj) > len(c.subj) + 1:
            c.obj = c.obj[len(c.subj) + 1:]
        if c.obj:
            neg_pol = c.obj in ("何も", "誰も")
            if neg_pol: c.neg = True
            out.append(c.obj + ("" if neg_pol else c.obj_particle if c.obj_particle is not None else ("が" if (c.verb and (c.verb[0] in GA_VERBS or c.modal == "can") and not c.want) else "を")))
        if c.wh and c.wh[1] == "o" and not c.copula:
            out.append(c.wh[0] + "を")
        if c.adv: out.append(c.adv)
        if c.embedded is not None:
            out.append(generate(c.embedded, polite=False, embedded=True)[:-1] + "と")
        if c.wh and c.copula is not None and c.copula[0] == "wh" and c.wh[1] in ("o", "s", "adv", "loc", "t", "count"):
            if embedded: out.append(c.wh[0] + ("だった" if c.past else "") + "か")
            else: out.append(Pred("na", c.wh[0]).finite(c.past, c.neg, polite) + "か")
        else:
            if c.copula is not None and c.copula[0] == "adv":
                out.append(c.copula[1])
            if c.infinitive is not None:
                mode, inner = c.infinitive
                body = generate(inner, polite=False, embedded=True)[:-1]
                lemma0, ja0, conj0 = inner.verb if inner.verb else (None, None, None)
                p0 = (make_pred(ja0, conj0) if conj0 not in ("fixed", "ia_fixed") else Pred("fixed", ja0)) if ja0 else None
                adjf = predicate_form(c, polite)
                ipred = predicate_form(inner, False)
                if mode == "extraposed":
                    if body.endswith("だ"): body = body[:-1] + "な"
                    out.append(body + "のは" + c.degree + adjf)
                elif mode == "adj_to":
                    out.append(body[:-len(ipred)] + (p0.te() if p0 and p0.kind != "fixed" else ipred) + c.degree + adjf)
                elif mode == "enough":
                    out.append(body + "のに十分" + adjf)
                elif mode == "too":
                    kind0, *rest0 = c.copula
                    a = adj_pred(rest0[0][0], rest0[0][1]) if kind0 == "adj" else None
                    base = (a.stem + "すぎて") if a is not None else ""
                    out.append(base + (p0.potential().finite(False, True, polite) if p0 and p0.kind in ("v1", "v5", "suru", "kuru") else "できない"))
            else:
                out.append(c.degree + predicate_form(c, polite if not c.suffix else False) + c.suffix)
            if c.question and not c.tag: out.append("か")
            if c.tag: out.append("ね")
    return "".join(out) + "。"

def translate(tokens):
    toks = merge_multiword(tokens)
    frame = Parser(toks).parse()
    return generate(frame)

def merge_multiword(tokens):
    tokens = [x for t in tokens for x in (("can", "'t") if t == "cannot" else (t,))]
    out, i = [], 0
    while i < len(tokens):
        for mw in MULTIWORD:
            n = mw.count(" ") + 1
            if " ".join(tokens[i:i + n]) == mw:
                out.append(mw); i += n; break
        else:
            out.append(tokens[i]); i += 1
    return out
