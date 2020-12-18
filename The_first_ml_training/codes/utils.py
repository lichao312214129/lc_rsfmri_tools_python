# 检查相关性
# plt.figure()
# coef = spearmanr(features_values)
# coef = coef[0]
# sns.heatmap(coef, cmap="RdBu_r", vmin=-1, vmax=1, annot=False, square=True, fmt=".2f")
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.xticks(np.arange(0,12)+0.5, header, rotation=90)
# plt.yticks(np.arange(0,12)+0.5, header, rotation=0)
# plt.tight_layout()
# plt.show()


# plt.figure()
# cm = confusion_matrix(targets_test, pred_test)
# sns.heatmap(cm, cmap="RdBu_r", annot=True, square=True, fmt=".2f")
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# # plt.xticks(np.arange(0,12)+0.5, header, rotation=90)
# # plt.yticks(np.arange(0,12)+0.5, header, rotation=0)
# plt.tight_layout()
# plt.show()