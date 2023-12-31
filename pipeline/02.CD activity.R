library('phyloseq')
library("ape")
library("Biostrings")
library("ggplot2")
library("ggtree")
library('tidytree')
library('ggpubr')

# open sample information
## https://github.com/kangdy9801/Diagnosis-of-Crohn-s-Disease-and-Ulcerative-Colitis-Using-the-Microbiome/blob/main/input%20file/01.SAM_CD.txt
sampledata <- read.table(file = "01.SAM_CD.txt",
                         header = T,
                         sep = '\t',
                         quote = "",
                         check.names = F,
                         stringsAsFactors = F,
                         row.names = 1)

sampledata$Group <- factor(sampledata$Group, levels=c("HC","CD"))
SAM = sample_data(sampledata, errorIfNULL = TRUE)

# oepn Tax file
## https://github.com/kangdy9801/Diagnosis-of-Crohn-s-Disease-and-Ulcerative-Colitis-Using-the-Microbiome/blob/main/input%20file/bracken.biom
biomfilename = "bracken.biom"
data <- import_biom(biomfilename, parseFunction=parse_taxonomy_default)

#taxmat_table
colnames(data@tax_table@.Data) <- c("Kingdom","Phylum","Class","Order","Family","Genus","Species")
taxmat <- data@tax_table@.Data
TAX = tax_table(taxmat)

bacteria_filiter = subset_taxa(TAX, Kingdom =="k__Bacteria")

#otu
## https://github.com/kangdy9801/Diagnosis-of-Crohn-s-Disease-and-Ulcerative-Colitis-Using-the-Microbiome/blob/main/input%20file/03.rarefy_CD.txt
otumat <- read.table(file = "03.rarefy_CD.txt",
                     header = T,
                     sep = '\t',
                     quote = "",
                     check.names = F,
                     stringsAsFactors = F,
                     row.names = 1)

class(otumat)
OTU = otu_table(otumat, taxa_are_rows = TRUE)

#merge
physeq <- phyloseq(OTU, bacteria_filiter, SAM)
physeq

#iden
sample_names(physeq)
rank_names(physeq)


#Class
patient <- phyloseq::subset_samples(
  physeq = physeq,
  Class %in% c("HC","Mild","Moderate"))
sample_data(patient)$Class <- factor(sample_data(patient)$Class, levels=c("HC","Mild","Moderate"))


##02. alpha diversity
library(microbiomeutilities)
library(ggpubr)

# change color of line 
mycols <- c("springgreen3","brown3","steelblue")

chao1_alpha_plot <- plot_diversity_stats(patient, group = "Class", 
                                         index = "chao1",
                                         label.format="p.format",
                                         group.colors = mycols,
                                         stats = TRUE)+
  scale_x_discrete("Class", labels = c("HC" = "HC \n(n = 50)","Mild" = "Mild \n(n = 100)","Moderate" = "Moderate \n(n = 55)","Severe" = "Severe \n(n = 18)"))+
  theme(legend.position="none",
        axis.text.x=element_text(angle=0,hjust=0.5,vjust=0.5,size=15),
        axis.text.y = element_text(size=18),
        axis.title.y = element_text(size=20))+scale_y_continuous(name = 'Chao1')

##03. beta diversity
pseq.rda <- ordinate(physeq = patient, method = "PCoA",distance ="bray", cca = "Class")
scores <- pseq.rda$vectors
pcoa_df <- data.frame(sample_id = rownames(scores), PCoA1 = scores[, 1], PCoA2 = scores[, 2])

ord_RDA <- plot_ordination(physeq = patient, ordination = pseq.rda, color = "Class")
ord_RDA <- ord_RDA + geom_point(size = 7) + 
  scale_color_manual(values=mycols) + stat_ellipse() + theme_classic2() +
  theme(axis.text.y = element_text(size=30),
        axis.text.x = element_text(size = 30),
        legend.title = element_text(size = 25),
        legend.text = element_text(size = 25),
        axis.title.y = element_text(size=30),
        axis.title.x = element_text(size = 30))+
  labs(x = "PCo1 [19.5%]", y = "PCo2 [14.7%]")
ord_RDA


library("vegan")
totumat <- t(otu_table(patient))
sample_meta <- sample_data(patient)
sample_meta <- data.frame(sample_meta)

adonis2(totumat~Class, data= sample_meta, permutations=9999, method="bray")

##04. relative abundance
## https://github.com/kangdy9801/Diagnosis-of-Crohn-s-Disease-and-Ulcerative-Colitis-Using-the-Microbiome/blob/main/input%20file/02.filter.txt
otumat <- read.table(file = "02.filter.txt",
                     header = T,
                     sep = '\t',
                     quote = "",
                     check.names = F,
                     stringsAsFactors = F,
                     row.names = 1)
OTU = otu_table(otumat, taxa_are_rows = TRUE)

#merge
data_phylo_filt <- phyloseq(OTU, bacteria_filiter, SAM)
data_phylo_filt

#Group
physeq_CD <- phyloseq::subset_samples(
  physeq = data_phylo_filt,
  Group %in% c("HC","CD"))
sample_data(patient)$Group <- factor(sample_data(patient)$Group, levels=c("HC","CD"))

#Class
patient <- phyloseq::subset_samples(
  physeq = physeq_CD,
  Class %in% c("HC","Mild","Moderate"))
sample_data(patient)$Class <- factor(sample_data(patient)$Class, levels=c("HC","Mild","Moderate"))

#01. data
library('microeco')
sample_meta <- sample_data(patient)
otu <- otu_table(patient)
tax <- tax_table(patient)

sample_meta <- data.frame(sample_meta)

dataset <- microtable$new(sample_table = sample_meta, otu_table = otu, tax_table = tax)
dataset

#02.phylum, genus, species level
# The groupmean parameter can be used to obtain the group-mean barplot.
t1 <- trans_abund$new(dataset = dataset, taxrank = "Phylum", ntaxa = 10, groupmean = "Class")
t2 <- trans_abund$new(dataset = dataset, taxrank = "Genus", ntaxa = 10, groupmean = "Class")
t3 <- trans_abund$new(dataset = dataset, taxrank = "Species", ntaxa = 10, groupmean = "Class")

#03. plot
p1 <- t1$plot_bar(others_color = "grey15", legend_text_italic = FALSE)
p2 <- p1 + theme_classic() +
  scale_x_discrete("Class", labels = c("HC" = "HC \n(n = 50)","Mild" = "Mild \n(n = 49)","Moderate" = "Moderate \n(n = 19)"))+
  theme(axis.text.y = element_text(size=25),
        axis.text.x = element_text(angle=0,hjust=0.5,vjust=0.5,size=25),
        legend.title = element_text(size = 25),
        legend.text = element_text(size = 15),
        axis.title.y = element_text(size=30), 
        axis.title.x = element_text(size = 20))

g1 <- t2$plot_bar(others_color = "grey15", legend_text_italic = FALSE)
g2 <- g1 + theme_classic() +
  scale_x_discrete("Class", labels = c("HC" = "HC \n(n = 50)","Mild" = "Mild \n(n = 49)","Moderate" = "Moderate \n(n = 19)"))+
  theme(axis.text.y = element_text(size=25),
        axis.text.x = element_text(angle=0,hjust=0.5,vjust=0.5,size=25),
        legend.title = element_text(size = 25),
        legend.text = element_text(size = 15),
        axis.title.y = element_text(size=30), 
        axis.title.x = element_text(size = 20))

s1 <- t3$plot_bar(others_color = "grey15", legend_text_italic = FALSE)
s2 <- s1 + theme_classic() +
  scale_x_discrete("Class", labels = c("HC" = "HC \n(n = 50)","Mild" = "Mild \n(n = 49)","Moderate" = "Moderate \n(n = 19)"))+
  theme(axis.text.y = element_text(size=25),
        axis.text.x = element_text(angle=0,hjust=0.5,vjust=0.5,size=25),
        legend.title = element_text(size = 25),
        legend.text = element_text(size = 15),
        axis.title.y = element_text(size=30), 
        axis.title.x = element_text(size = 20))

##05. CD differential abudnacne
#  Linear (Lin) Model for Differential Abundance (DA) Analysis
library("MicrobiomeStat")

#00. data
physeq_CD
#02. linda
linda.obj <- linda(phyloseq.obj = physeq_CD,
                   feature.dat.type = "count", 
                   p.adj.method = "fdr",
                   alpha = 0.05,
                   is.winsor = TRUE, 
                   formula = '~Group',
                   zero.handling = "pseudo-count", 
                   pseudo.cnt = 1) # pseudo coun 
res <- linda.obj$output
res$GroupCD

#03.data.frame
linda_res_df <- data.frame(
  Species = row.names(res$GroupCD),
  baseMean = unlist(res$GroupCD$baseMean),
  logFC = unlist(res$GroupCD$log2FoldChange),
  lfcSE = unlist(res$GroupCD$lfcSE), # standard errors
  stat = unlist(res$GroupCD$stat),#log2FoldChange / lfcSE
  pvalue = unlist(res$GroupCD$pvalue),
  padj = unlist(res$GroupCD$padj),
  reject = unlist(res$GroupCD$reject),# padj <= alpha
  df = unlist(res$GroupCD$df)) #degrees of freedom
linda_res_df[1:5, 1:9]

#04. plot
library(EnhancedVolcano)
EnhancedVolcano(linda_res_df,
                lab = linda_res_df$Species,
                x = 'logFC',
                y = 'padj',
                subtitle = bquote(italic('FDR <= 0.05, absolute log2FC >= 1')),
                pCutoff= 0.05,
                FCcutoff = 1,
                pointSize = 4,
                labSize=5,
                selectLab=c(""),
                xlab = bquote(~Log[2]~ 'fold change'),
                labCol = 'black',
                labFace = 'bold',
                ylim = c(0, max(-log10(linda_res_df[["pvalue"]]), na.rm = TRUE) + 1),
                boxedLabels = TRUE,
                colAlpha = 5/5,
                legendPosition = 'none',
                legendLabSize = 14,
                legendIconSize = 4.0,
                drawConnectors = TRUE,
                widthConnectors = 1.0,
                colConnectors = 'black')


##06. Taxon set enrichment analsis
library(RColorBrewer)

#01.TSEA_list 
## https://github.com/kangdy9801/Diagnosis-of-Crohn-s-Disease-and-Ulcerative-Colitis-Using-the-Microbiome/blob/main/input%20file/cd_disease.txt
TSEA_list <- read.table(file = "cd_disease.txt",
                        header = T,
                        sep = '\t',
                        quote = "",
                        check.names = F,
                        stringsAsFactors = F)
head(TSEA_list)
TSEA_list$Tendency <- factor(TSEA_list$Tendency, levels = c("Increase","Decrease"))

#02. color
mypalette <- brewer.pal(3, "YlGnBu")

#03. plot
tsea_plot <- ggplot(data = TSEA_list)+
  geom_point(aes(x = reorder(From_Curated_Host_intrinsic_Taxon_Sets, hits),
                 y = hits,
                 color = Tendency,
                 size = p.value))+
  coord_flip()+
  theme_bw() +
  theme_classic() +
  labs(x = "From Curated Host-intrinsic Taxon Set",
       y = "Hits",
       title = "TSEA") +
  theme(axis.text.y = element_text(size = 13),
        axis.text.x=element_text(size=13),
        axis.title.x = element_text(size=13),
        axis.title.y = element_text(size=13),
        legend.text = element_text(size = 13), legend.title = element_text(size = 13))
tsea_plot

##07. CD activity differential abundance
#Class
patient <- phyloseq::subset_samples(
  physeq = physeq_CD,
  Class %in% c("Mild","Moderate"))
sample_data(patient)$Class <- factor(sample_data(patient)$Class, levels=c("Mild","Moderate"))


#05. linda
linda.obj <- linda(phyloseq.obj = patient,
                   feature.dat.type = "count", 
                   p.adj.method = "fdr",
                   alpha = 0.1,
                   is.winsor = TRUE,  
                   formula = '~Class', 
                   zero.handling = "pseudo-count", 
                   pseudo.cnt = 1) # pseudo coun 
res <- linda.obj$output
res$ClassModerate

#06.data.frame
linda_res_df <- data.frame(
  Species = row.names(res$ClassModerate),
  baseMean = unlist(res$ClassModerate$baseMean),
  logFC = unlist(res$ClassModerate$log2FoldChange),
  lfcSE = unlist(res$ClassModerate$lfcSE), # standard errors
  stat = unlist(res$ClassModerate$stat),#log2FoldChange / lfcSE
  pvalue = unlist(res$ClassModerate$pvalue),
  padj = unlist(res$ClassModerate$padj),
  reject = unlist(res$ClassModerate$reject),# padj <= alpha
  df = unlist(res$ClassModerate$df)) #degrees of freedom
linda_res_df[1:5, 1:9]

#07. plot
library(EnhancedVolcano)
EnhancedVolcano(linda_res_df,
                lab = linda_res_df$Species,
                x = 'logFC',
                y = 'pvalue',
                subtitle = bquote(italic('p-value <= 0.001, absolute log2FC >= 1')),
                pCutoff= 0.001,
                FCcutoff = 1,
                pointSize = 4,
                labSize=5,
                selectLab=c("328814","182210","1743168"),
                xlab = bquote(~Log[2]~ 'fold change'),
                labCol = 'black',
                labFace = 'bold',
                ylim = c(0, max(-log10(linda_res_df[["pvalue"]]), na.rm = TRUE) + 1),
                boxedLabels = TRUE,
                colAlpha = 5/5,
                legendPosition = 'none',
                legendLabSize = 14,
                legendIconSize = 4.0,
                drawConnectors = TRUE,
                widthConnectors = 1.0,
                colConnectors = 'black')

##08. bar plot
#01.list
list <- c("328814","182210","1743168")

#02.plot
for(taxa in list){
  df2_taxa <- data.frame(Abundance = abundances(x = patient, transform = "clr")[taxa,],
                         Class = meta(patient)$Class)
  
  #03. color
  mycols <- c("springgreen3","brown3","steelblue","goldenrod2")
  names(mycols) <- levels(df2_taxa$Class)
  
  custom_colors <- scale_colour_manual(values = mycols)
  
  #04. class
  class <- c("HC","Mild","Moderate")
  class_pairs <- combn(seq_along(class), 2, simplify = F , FUN = function(i)class[i])
  
  #05. ggplot
  p2_df2_taxa <- ggplot(df2_taxa, aes(x = Class, y = Abundance, fill = Class))+
    scale_fill_manual(values = c("HC" ="springgreen3",
                                 "Mild" = "brown3",
                                 "Moderate" = "steelblue")) +
    geom_boxplot()+ 
    theme_bw()+ 
    theme_classic() +
    labs(title = taxa, y = "CLR abundances")+ 
    
    scale_x_discrete("Class", labels = c("HC" = "HC \n(n = 50)","Mild" = "Mild \n(n = 40)","Moderate" = "Moderate \n(n = 19)"))+
    theme(plot.title = element_text(size=13))+
    theme(legend.position="none",
          axis.text.x=element_text(angle=0,hjust=0.5,vjust=0.5,size=15),axis.title.x = element_blank(),
          axis.text.y = element_text(size=18),
          axis.title.y = element_text(size=18))
  
  p3 <- p2_df2_taxa + stat_compare_means(comparisons = class_pairs, label = "p.signif", method = "t.test")}