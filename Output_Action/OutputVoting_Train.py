# 逻辑回归分类器
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression()
#  随机森林分类器
rnd_clf = RandomForestClassifier()
# 支持向量机分类器
svm_clf = SVC(probability=True)
# 集成以上3种分类器的投票分类器

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft'
)
