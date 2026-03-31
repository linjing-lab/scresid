from .reg_plot import draw_reg
import scanpy

def evaluate(eval_adata, cell_type, key_dic):
    print("PCA cluster results...")
    scanpy.tl.pca(eval_adata) # default pca result
    scanpy.pl.pca(eval_adata, color=key_dic['condition_key'], frameon=False, title="PCA of " + cell_type + " by Condition")
    print("Compute rank_genes_groups...")
    scanpy.tl.rank_genes_groups(eval_adata, groupby=key_dic['condition_key'], reference=key_dic['ctrl_key'], method="wilcoxon")
    degs_pred = eval_adata.uns["rank_genes_groups"]["names"][key_dic['pred_key']]
    degs_ctrl = eval_adata.uns["rank_genes_groups"]["names"][key_dic['stim_key']]
    print("Select common DEGS...")
    common_degs = list(set(degs_ctrl[0:100]) & set(degs_pred[0:100]))
    common_nums = len(common_degs)
    print("Common DEGs: ", common_nums)
    print("Regression plot of mean...")
    res_mean = draw_reg(eval_adata=eval_adata,
                        cell_type=cell_type,
                        reg_type='mean',
                        axis_keys={"x": "predict", "y": key_dic['stim_key']},
                        condition_key=key_dic['condition_key'],
                        text_gene=degs_ctrl[:10],
                        top_genes=degs_ctrl[:100],
                        save_path=None,
                        title=None,
                        draw=True,
                        if_fig=False,
                        fontsize=12
                        )
    print("R2 mean of all genes {}, top genes {}".format(res_mean[0], res_mean[1]))
    print("Regression plot of var...")
    res_var = draw_reg(eval_adata=eval_adata,
                       cell_type=cell_type,
                       reg_type='var',
                       axis_keys={"x": "predict", "y": key_dic['stim_key']},
                       condition_key=key_dic['condition_key'],
                       text_gene=degs_ctrl[:10],
                       top_genes=degs_ctrl[:100],
                       save_path=None,
                       title=None,
                       draw=True,
                       if_fig=False,
                       fontsize=12
                       )
    print("R2 var of all genes {}, top genes {}".format(res_var[0], res_var[1]))
    gene = degs_ctrl[0]
    print(f"Plot violin graph of first gene {gene}...")
    scanpy.pl.violin(eval_adata, keys=gene, groupby=key_dic['condition_key'])
    gene = degs_ctrl[1]
    print(f"Plot violin graph of second gene {gene}...")
    scanpy.pl.violin(eval_adata, keys=gene, groupby=key_dic['condition_key'])
    gene = degs_ctrl[2]
    print(f"Plot violin graph of third gene {gene}...")
    scanpy.pl.violin(eval_adata, keys=gene, groupby=key_dic['condition_key'])
    print("Compute DEGS contrast...")
    scanpy.tl.rank_genes_groups(eval_adata, groupby=key_dic['condition_key'], reference=key_dic['ctrl_key'], method="wilcoxon")
    print("Plot 25 genes of eval_adata...")
    scanpy.pl.rank_genes_groups(eval_adata, n_genes=25, sharey=False, show=True)

    marker_genes = degs_ctrl[0:20]
    print(f"Dotplot of the first 20 genes...")
    scanpy.pl.dotplot(eval_adata, marker_genes, groupby=key_dic['condition_key'], show=True)