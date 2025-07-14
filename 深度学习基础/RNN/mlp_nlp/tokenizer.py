from transformers import AutoTokenizer
import matplotlib.pyplot as plt

tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")

# 分词效果示例，三段文本表示的意思是相近的
text_fr = '''Évariste Galois (/ɡælˈwɑː/; français : [evaʁist ɡalwa] ; 25 octobre 1811 - 31 mai 1832) était un mathématicien français et un militant politique. Alors qu'il était encore adolescent, il parvint à déterminer une condition nécessaire et suffisante pour qu'un polynôme soit résoluble par des radicaux, résolvant ainsi un problème qui était resté ouvert pendant 350 ans. Son travail posa les fondements de la théorie de Galois et de la théorie des groupes, deux branches majeures de l'algèbre abstraite. Il était un fervent républicain et fut très impliqué dans les troubles politiques qui entourèrent la Révolution française de 1830. En raison de son activisme politique, il fut arrêté à plusieurs reprises, purgé une peine de plusieurs mois de prison. Pour des raisons restées obscures, peu de temps après sa libération de prison, il se battit en duel et décéda des blessures qu'il subit.'''
text_en = '''Évariste Galois (/ɡælˈwɑː/; French: [evaʁist ɡalwa]; 25 October 1811 – 31 May 1832) was a French mathematician and political activist. While still in his teens, he was able to determine a necessary and sufficient condition for a polynomial to be solvable by radicals, thereby solving a problem that had been open for 350 years. His work laid the foundations for Galois theory and group theory, two major branches of abstract algebra. He was a staunch republican and was heavily involved in the political turmoil that surrounded the French Revolution of 1830. As a result of his political activism, he was arrested repeatedly, serving one jail sentence of several months. For reasons that remain obscure, shortly after his release from prison he fought in a duel and died of the wounds he suffered.'''
text_zh = '''埃瓦里斯特·伽罗瓦（法语：Évariste Galois，1811年10月25日—1832年5月31日，法语发音： [evaʁist ɡalwa]）是一位法国数学家和政治活动家。尽管还在十几岁时，他就能够确定多项式能够通过根式求解的充分必要条件，从而解决了一个悬而未决的问题，该问题已经存在了350年。他的工作奠定了Galois理论和群论的基础，这两个是抽象代数的重要分支。他是一位坚定的共和派，深度参与了1830年法国大革命期间的政治动荡。由于他的政治活动，他多次被逮捕，其中一次入狱数月。由于原因不明，他在刑满释放后不久，参与了一场决斗并因受伤而去世。'''

texts={
    'fr':text_fr,
    'en':text_en,
    'zh':text_zh
}

def get_token_stats(tokenizer):
    # 统计文本中的单词数量(如果是中文，则为文本的字数)
    str_stats={}
    # 统计分词后的词元数量
    token_stats={}
    for (k,v) in texts.items():
        text_len=len(v.split()) if k!='zh' else len(v)
        token_len=len(tokenizer.encode(v))
        str_stats[k]=text_len
        token_stats[k]=token_len
    print(str_stats)
    print(token_stats)
    return draw_bar(str_stats, token_stats)

def draw_bar(str_stats, token_stats):
    # 将统计结果可视化
    fig=plt.figure(figsize=(6,6),dpi=80)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 13})
    bar_width=0.1
    base=range(len(str_stats))
    br_str=[x-bar_width for x in base]
    br_token=[x+bar_width for x in base]
    plt.bar(br_str, str_stats.values(), color='g',
            width=bar_width * 2, label='文本长度')
    plt.bar(br_token, token_stats.values(), color='b',
            width=bar_width * 2, label='分词后的长度')
    plt.xticks([r for r in base], str_stats.keys(), fontsize=18)
    plt.legend(shadow=True)
    return fig

from datasets import load_dataset

# 使用中文预料训练分词器
raw_data = load_dataset('BelleGroup/train_0.5M_CN')

def get_training_corpus():
    # 为了减少运算时间，只选择较少的训练数据
    data=raw_data['train'].select(range(10000))
    for idx in range(0,len(data),1000):
        samples=data[idx:idx+1000]
        yield samples.get('instruction', []) + samples.get('output', [])

