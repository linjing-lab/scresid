import seaborn, numpy, pandas
import matplotlib.pyplot as plt
from adjustText import adjust_text

def draw_reg(eval_adata, # adata of gene expression containing control, stimulated, predict
             cell_type, # cell type of the data
             reg_type='mean', # mean or var
             axis_keys={"x": "predict", "y": "stimulated"},
             condition_key='condition',
             text_gene=None, # a list of genes which you want to color red
             top_genes=None, # a list of top DEGs whose regression coefficients are to be calculated
             save_path=None, 
             title=None, # the title of the the regression figure
             draw=True,
             if_fig=False,
             fontsize=14
             ):
    '''
    Plot the regression line between the predicted perturbation response and the true response
    '''
    df_stim = eval_adata[(eval_adata.obs[condition_key] == axis_keys["y"])].to_df()
    df_pred = eval_adata[(eval_adata.obs[condition_key] == axis_keys["x"])].to_df()
    if reg_type == 'mean':
        stim_value = df_stim.mean().values.reshape(-1, 1)
        pred_value = df_pred.mean().values.reshape(-1, 1)
    elif reg_type == 'var':
        stim_value = df_stim.var().values.reshape(-1, 1)
        pred_value = df_pred.var().values.reshape(-1, 1)
    else:
        raise ValueError(f"regression type supports mean and var, {reg_type} not support.")
    data = numpy.hstack((stim_value, pred_value))
    data_df = pandas.DataFrame(data, columns=['case', 'predict'], index=df_stim.columns)
    fig, ax = plt.subplots()
    seaborn.set(color_codes=True)
    seaborn.regplot(x='case', y='predict', data=data_df, ax=ax)
    if text_gene is not None:
        texts = []
        case = stim_value
        predict = pred_value
        for i in text_gene:
            j = eval_adata.var_names.tolist().index(i)
            case_bar = case[j]
            predict_bar = predict[j]
            texts.append(plt.text(case_bar, predict_bar, i, fontsize=11, color="black"))
            ax.plot(case_bar, predict_bar, "o", color="red", markersize=5)
        adjust_text(texts, x=case, y=predict, ax=ax, arrowprops=dict(arrowstyle="->", color="grey", lw=0.5), force_points=(0.0, 0.0),)
    if top_genes is not None:
        data_deg = data_df.loc[top_genes, :]
        r_top = round(data_deg['case'].corr(data_deg['predict'], method='pearson'), 3)
        x_scale = 0.1 * numpy.max(data_df['case'])
        y_scale = 0.85 * numpy.max(data_df['predict'])
        ax.text(x_scale, y_scale, s='$R^2_{top 100 genes}$=' + str(round(r_top * r_top, 3)), fontsize=fontsize, color='black')
    r = round(data_df['case'].corr(data_df['predict'], method='pearson'), 3)
    x_scale = 0.1 * numpy.max(data_df['case'])
    y_scale = 0.75 * numpy.max(data_df['predict'])
    ax.text(x_scale, y_scale, s='$R^2_{all genes}$=' + str(round(r * r, 3)), fontsize=fontsize, color='black')
    if title:
        plt.title(title)
    else:
        plt.title('Regression ' + reg_type + ' of ' + cell_type)
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    if draw:
        plt.show()
        plt.close()
    if if_fig:
        return [round(r * r, 3), round(r_top * r_top, 3), fig]
    else:
        return [round(r * r, 3), round(r_top * r_top, 3)]