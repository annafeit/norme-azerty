# Optimizing the placement of special characters on the physical keyboard

This code and data were used in the design process for developing the new French keyboard standard, <a href="https://normalisation.afnor.org/actualites/faq-clavier-francais/"> published in April 2019 by AFNOR</a>, the French standardization organization.

You can find more information on the resulting layout and the process at <a href=http://norme-azerty.fr>norme-azerty.fr</a> and in the following paper:

Anna Maria Feit, Mathieu Nancel, Maximilian John, Andreas Karrenbauer, Daryl Weir, and Antti Oulasvirta. (2021) <i>Azerty amélioré: Computational design on a national scale.</i> Communications of the ACM (CACM), to appear.

You are free to use and modify this code. If you do so, please cite the paper above.

We publish this code to facilitate the optimization of other keyboard layouts and languages. While it was used for French, you can use it to optimize layouts for any other language, provided that you have the right inpout data (language statistics, character sets, etc.). 
The jupyter notebook "Introduction to the code base.ipynb" introduces the available methods and explains how to customize it for your own character set or language. 

System requirements: 
Python 3 with packages numpy, matplotlib, pandas, codecs, os, unicodedata, jupyter
<a href="https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_python_installation_opt.html"> Gurobi solver with gurobipy</a>
