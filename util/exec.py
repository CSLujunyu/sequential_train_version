import jieba
jieba.load_userdict('./THUOCL_food_userdict.txt')

print(list(jieba.cut('我喜欢抹茶香饮料')))
print(list(jieba.cut('我喜欢奥利奥饮料')))
print(list(jieba.cut('抹茶流心卷外面卷子是抹茶的里面夹心是奶油')))


### prepare data process

#prepare_data  -> build_corpus  -> train_glove  -> preprocessing