library('phyloseq')
library("ape")
library("Biostrings")
library("ggplot2")
library("ggtree")
library('tidytree')
library('ggpubr')

##01.update data
# open sample infromation
sampledata <- read.table(file = "00.SAM.txt",
                         header = T,
                         sep = '\t',
                         quote = "",
                         check.names = F,
                         stringsAsFactors = F,
                         row.names = 1)
## https://github.com/kangdy9801/Diagnosis-of-Crohn-s-Disease-and-Ulcerative-Colitis-Using-the-Microbiome/blob/main/input%20file/00.SAM.txt

sampledata$Group <- factor(sampledata$Group, levels=c("HC","CD","UC"))
SAM = sample_data(sampledata, errorIfNULL = TRUE)

# oepn Tax file
## https://github.com/kangdy9801/Diagnosis-of-Crohn-s-Disease-and-Ulcerative-Colitis-Using-the-Microbiome/blob/main/input%20file/bracken.biom
biomfilename = "bracken.biom"
data <- import_biom(biomfilename, parseFunction=parse_taxonomy_default)

colnames(data@tax_table@.Data) <- c("Kingdom","Phylum","Class","Order","Family","Genus","Species")
taxmat <- data@tax_table@.Data
TAX = tax_table(taxmat)

bacteria_filiter = subset_taxa(TAX, Kingdom =="k__Bacteria")

otumat <- data@otu_table@.Data

class(otumat)
OTU = otu_table(otumat, taxa_are_rows = TRUE)

#merge
physeq <- phyloseq(OTU, bacteria_filiter, SAM)
physeq

#iden
sample_names(physeq)
rank_names(physeq)

# remove low abundance
data_phylo_filt = filter_taxa(physeq, function(x) sum(x > 4) > (0.05 * length(x)), TRUE) 

# rarefy even depth 
set.seed(2023)
data_phylo_filt_rarefy <- rarefy_even_depth(physeq = data_phylo_filt,
                                            sample.size = min(sample_sums(data_phylo_filt)), 
                                            replace = F, 
                                            trimOTUs = T)

##02. alpha diversity
library(microbiomeutilities)
library(ggpubr)

# change color of line 
mycols <- c("springgreen3","brown3","steelblue")

comps <- make_pairs(sample_data(data_phylo_filt_rarefy)$Group)
print(comps)

# chao
chao1_alpha_plot <- plot_diversity_stats(data_phylo_filt_rarefy, 
                                         dot.opacity = 0.3,
                                         box.opacity = 0.3,
                                         violin.opacity = 0.6,
                                         group = "Group", 
                                         index = "chao1",
                                         # label.format="p.signif",
                                         group.colors = mycols,
                                         stats = F)+
  scale_x_discrete("Group", labels = c("HC" = "HC \n(n = 50)","CD" = "CD \n(n = 173)","UC" = "UC \n(n = 259)"))+
  theme(legend.position="none", axis.title.x=element_blank(),
        axis.text.x=element_text(angle=0,hjust=0.5,vjust=0.5,size=25),
        axis.text.y = element_text(size=25),
        axis.title.y = element_text(size=25))+scale_y_continuous(name = 'Chao1')
p <- chao1_alpha_plot + stat_compare_means(
  comparisons = comps,
  paired = F, size = 7,
  label = "p.format",
  tip.length = 0.05,
  method = "wilcox.test")
p

##03. beta diversity
pseq.rda <- ordinate(physeq = data_phylo_filt_rarefy, method = "PCoA",distance ="bray", cca = "Group")

scores <- pseq.rda$vectors
pcoa_df <- data.frame(sample_id = rownames(scores), PCoA1 = scores[, 1], PCoA2 = scores[, 2])

ord_RDA <- plot_ordination(physeq = data_phylo_filt_rarefy, ordination = pseq.rda, color = "Group")
ord_RDA <- ord_RDA + geom_point(size = 7) + 
  scale_color_manual(values=mycols) + stat_ellipse() + theme_classic2() +
  theme(axis.text.y = element_text(size=30),
        axis.text.x = element_text(size = 30),
        legend.title = element_text(size = 25),
        legend.text = element_text(size = 25),
        axis.title.y = element_text(size=30),
        axis.title.x = element_text(size = 30))+
  labs(x = "PCo1 [18.1%]", y = "PCo2 [14.3%]")
ord_RDA


#03. permanova
library("vegan")
totumat <- t(otu_table(data_phylo_filt_rarefy))
sample_meta <- sample_data(data_phylo_filt_rarefy)
sample_meta <- data.frame(sample_meta)

pvalue <- adonis2(totumat~Group, data= sample_meta, permutations=9999, method="bray")

##04. relative abundance
library('microeco')

#01. input
sample_meta <- sample_data(data_phylo_filt)
otu <- otu_table(data_phylo_filt)
tax <- tax_table(data_phylo_filt)

dataset <- microtable$new(sample_table = sample_meta, otu_table = otu, tax_table = tax)
dataset

#02. Species level, Genus level, phylum level
t1 <- trans_abund$new(dataset = dataset, taxrank = "Species", ntaxa = 10, groupmean = "Group")
t2 <- trans_abund$new(dataset = dataset, taxrank = "Genus", ntaxa = 10, groupmean = "Group")
t3 <- trans_abund$new(dataset = dataset, taxrank = "Phylum", ntaxa = 10, groupmean = "Group")

#03. plot
s1 <- t1$plot_bar(others_color = "grey15", legend_text_italic = FALSE)
s2 <- s1 + theme_classic() +
  scale_x_discrete("Group", labels = c("HC" = "HC \n(n = 50)","CD" = "CD \n(n = 173)","UC" = "UC \n(n = 259)"))+
  theme(axis.text.y = element_text(size=15),
        axis.text.x = element_text(angle=0,hjust=0.5,vjust=0.5,size=15),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8),
        axis.title.y = element_text(size=15), 
        axis.title.x = element_blank())

g1 <- t2$plot_bar(others_color = "grey15", legend_text_italic = FALSE)
g2 <- g1 + theme_classic() +
  scale_x_discrete("Group", labels = c("HC" = "HC \n(n = 50)","CD" = "CD \n(n = 173)","UC" = "UC \n(n = 259)"))+
  theme(axis.text.y = element_text(size=15),
        axis.text.x = element_text(angle=0,hjust=0.5,vjust=0.5,size=15),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8),
        axis.title.y = element_text(size=15), 
        axis.title.x = element_blank())

p1 <- t3$plot_bar(others_color = "grey15", legend_text_italic = FALSE)
p2 <- p1 + theme_classic() +
  scale_x_discrete("Group", labels = c("HC" = "HC \n(n = 50)","CD" = "CD \n(n = 173)","UC" = "UC \n(n = 259)"))+
  theme(axis.text.y = element_text(size=15),
        axis.text.x = element_text(angle=0,hjust=0.5,vjust=0.5,size=15),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 8),
        axis.title.y = element_text(size=15), 
        axis.title.x = element_blank())


##05. Differential Abundance
#Linear (Lin) Model for Differential Abundance (DA) Analysis
library("MicrobiomeStat")

#00. data
data_phylo_filt

#01. group 
# pairwise comparison between CD and UC
patient <- phyloseq::subset_samples(
  physeq = data_phylo_filt, 
  Group %in% c("CD","UC"))

#02. linda transfor CLR
linda.obj <- linda(phyloseq.obj = patient,
                   feature.dat.type = "count", #data type > count
                   p.adj.method = "fdr",
                   alpha = 0.05,
                   is.winsor = TRUE, 
                   formula = '~Group', 
                   zero.handling = "pseudo-count", 
                   pseudo.cnt = 1) 
res <- linda.obj$output

#04.data.frame
linda_res_df <- data.frame(
  Species = row.names(res$GroupUC),
  baseMean = unlist(res$GroupUC$baseMean),
  logFC = unlist(res$GroupUC$log2FoldChange),
  lfcSE = unlist(res$GroupUC$lfcSE), # standard errors
  stat = unlist(res$GroupUC$stat),#log2FoldChange / lfcSE
  pvalue = unlist(res$GroupUC$pvalue),
  padj = unlist(res$GroupUC$padj),
  reject = unlist(res$GroupUC$reject),# padj <= alpha
  df = unlist(res$GroupUC$df)) #degrees of freedom
linda_res_df[1:5, 1:9]

#05. plot
library(EnhancedVolcano)
EnhancedVolcano(linda_res_df,
                lab = linda_res_df$Species,
                x = 'logFC',
                y = 'padj',
                title="Linear (Lin) Model for Differential Abundance (DA) Analysis \n between CD and UC",
                subtitle = bquote(italic('FDR <= 0.05, absolute log2FC >= 1')),
                pCutoff= 0.05,
                FCcutoff = 1,
                pointSize = 4,
                labSize=5,
                selectLab=c(""),
                xlab = bquote(~Log[2]~ 'fold change'),
                labCol = 'black',
                labFace = 'bold',
                boxedLabels = TRUE,
                colAlpha = 5/5,
                legendPosition = 'none',
                legendLabSize = 14,
                legendIconSize = 4.0,
                drawConnectors = TRUE,
                widthConnectors = 1.0,
                colConnectors = 'black')


##06. Taxon set enrichment analysis
library(RColorBrewer)

#01.TSEA_list 
## https://github.com/kangdy9801/Diagnosis-of-Crohn-s-Disease-and-Ulcerative-Colitis-Using-the-Microbiome/blob/main/input%20file/disease.txt
TSEA_list <- read.table(file = "disease.txt",
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