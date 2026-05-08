# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, RAYLEIGH RESEARCH OY. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

""" Result automation compiler """

from glob import glob
import os
import re
import subprocess
import sys
import pandas as pd
from TexSoup.data import *
from TexSoup import TexSoup
import xml.etree.ElementTree as ET

def generate_tex_with_texsoup(template, figures_paths,xml,log_dir,
                              output_tex_file,
                              images_to_include = ['histogram','autocorrelation','midquote','return','slope',
                                                    'relative','volume', 'trade','trade-return'],
                              image_range= (2000,4000),
                              additional_metrics=None,
                              title='Rayleigh Research QR report',
                              author='RR QR team'):
    '''
    TODO: edit this document
    params:
    template: file path to existing latex document, if this doesn't exist there is small template in this file
    figures_paths: absolute or relative paths to image files. Document will be placed there so then we can get the only 'relative' locs
    output_tex_file: file path
    images_to_include: Figures that are inserted into the report, checking is done buy contains level, if None we will include all
    image_range: If figures contain numeric value at the end this will be the tuple of min and max range
    title: Report title
    author: Report author(s)
    '''
    # additional metrics should be parsed from csv to dict
    # Check if the folder exists
    if not os.path.exists(template):
        # This is unfinished
        print(f'[ERROR]: Template not found at {template}') 
        document = DEFAULT_SOUP
    else:
        with open(template,'r') as temp_file:
            document = TexSoup(temp_file.read()) 
    try:
        bookId = int(os.path.split(os.path.split(figures_paths[0])[0])[1].split('_')[-1])
    except:
        print('Book id parse error')
        bookId = 0
    # Parse simul parameter out config
    try:
        simul_params = extract_parameters(xml)
    except Exception as e:
        print('Could not parse xml')
        print(f'Error message {e}')
        simul_params = {}


    if len(figures_paths) == 0:
        print(f"No valid image files found.")
        return

    # Example tabular 'item' append
    tabular = document.find('tabularx')
    for key,value in simul_params.items():
        tabular = tab_append_row(tabular,key,value)

    # Load metrics
    if log_dir:
        metrics = {'Lag1':0,'GARCH': {'Beta':0, 'Alpha':0}, 'cointegration': {'TraceVal':0,'CriticalVal':0},
                'Volume':{'Sum':[0], 'Count':[0], 'Mean':[0], 'MeanTime':[0], 'CountTime':[0],'TradePeriod':[0]}, 
                'VPIN':{'VPIN':0, 'TradePeriod':0}, 'ParkVol':{'ParkVol':0, 'TradePeriod':0} }
        search_patterns = {'GARCH':'betaalpha', 'ParkVol':'parkinson','Volume':'trade_volume_period'}
        for name in metrics.keys():
            if name in search_patterns:
                extract_metrics(log_dir,bookId,metrics,name,subfields=metrics[name].keys(),search_pattern=f'*{search_patterns[name]}*.csv')
            else:
                subfields= metrics[name].keys() if type(metrics[name]) == dict else None
                extract_metrics(log_dir,bookId,metrics,name,subfields=subfields,search_pattern=f'*{name.lower()}*.csv')

        for key,values in metrics.items():
            if type(values) == dict:
                if key == 'Volume':
                    add_metrics_table(document,values,key,[' ','Volumes','Trade period'],n_cols=3)
                else:
                    add_metrics_table(document,values,key,list(values.keys()),len(values))
            else:
                add_metrics_table(document,values,key)
    
    
    if type(additional_metrics) == dict:
        for key,values in additional_metrics.items():
            add_metrics_table(document,values,key)
    # TODO if passed only path to csv (or csvs)
    # elif type(additional_metrics) == str:
        # add_met = {}
        # extract_metrics(additional_metrics,bookId,add_met,)
    # Add each image to the document
    image_set = set() # don't take two of the same images
    fig_idx = 0
    # NOTE if we do not include images, we will take all
    if images_to_include:
        for image_type in images_to_include:
            tmp_images=[]
            for tmp in figures_paths:
                if image_type in tmp:
                    img_num = os.path.split(tmp)[-1].split('.')[0].split('_')[-1]
                    if img_num.isnumeric():
                        img_num = int(img_num)
                        if  img_num < image_range[0] or img_num> image_range[1]:
                            continue
                    if image_set.__contains__(tmp):
                        continue
                    else:
                        image_set.add(tmp)
                        tmp_images.append(tmp)       
            fig_idx = fill_in_figures(document,tmp_images,fig_idx, image_type)
    else:
        fig_idx = fill_in_figures(document,figures_paths,fig_idx,'Aggregation plots')
    
    
    document.title.string = title # f'Simulation report Aggregated'
    document.author.string= author 
    
    
    # Write the generated LaTeX content to the output file
    try:
        with open(output_tex_file, "w") as tex_file:
            tex_file.write(str(document))
        print(f"LaTeX file successfully created at: {output_tex_file}")
    except Exception as e:
        print(f"Error writing to the file: {e}")

def tab_append_row(tabular,key=None,values=None):
    if key==None:
        for _,value in values.items():    
          tabular.append(f'{value} ')
          tabular.append('& ')
        tabular.contents.pop(len(tabular.contents)-1)
    else:
        tabular.append(f'{key} ')
        tabular.append(f' & {values}')
    tabular.append('\t')
    tabular.append('\\\\ ')
    tabular.append('\t')
    tabular.append(TexCmd('hline'))
    tabular.append('\n')
    return tabular


def add_metrics_table(document, values, key='met',titles=['Metrics','Value'], n_cols=2):
    hline = TexCmd('hline')
    top_bar = ['\n',hline,'\n']
    named_columns = [TexCmd('textbf',args=[BraceGroup(title)]) for title in titles]
    col_separators = [' & ' for _ in range(n_cols)]
    columns = list(sum(zip(named_columns,col_separators),()))
    columns.pop()
    first_line = top_bar + columns +  ['\t',r"""\\""",'\t', hline,'\n']
    if key == 'Lag1':
        first_line = [hline,'\n']
    tab_col_setting = BraceGroup("".join(['|'] + ['X|' for _ in range(n_cols)]))

    tabular = TexNamedEnv('tabularx', contents=first_line, args=[BraceGroup(TexCmd('columnwidth')),tab_col_setting])
    table = TexNamedEnv('table',contents=[TexCmd('centering'),'\n'],args=[BracketGroup('!htpb')])
    table.append(tabular)


    if type(values) == dict:
        if key == 'Volume':
            for t_key,value in values.items():
                if t_key != 'TradePeriod':
                    tabular.append(f'\t {t_key} ')
                    #print(f'{t_key=}|{value=}|{values=}')
                    #for t_period,t_value in value.items():
                    for t_period,t_value in enumerate(value):
                        if type(t_value) == float:
                            t_value = "{:.2f}".format(t_value)
                        tabular.append(f'\t & {t_value} &  {t_period} \\\\ \n') 
                    tabular.append(TexCmd('hline'))
                    tabular.append('\n')
        else:
            tabular = tab_append_row(tabular,None,values)
    else:
        tabular = tab_append_row(tabular,key,values)
    table.append('\n\t')
    table.append(TexCmd('caption', args=[BraceGroup(f'{key}')]))
    table.append('\n\t')
    table.append(TexCmd('label',args=[BraceGroup(f'tab:{key}')]))
    table.append('\n')
    document.document.append(table)
    document.document.append('\n')

def extract_metrics(log_dir, bookId, metrics, name, subfields=None, search_pattern=None):
    if search_pattern == None:
        search_pattern= f'*{name.lower()}*.csv'
    dirs = glob(os.path.join(log_dir,search_pattern))
    for csv_file in dirs:
        try:
            df = pd.read_csv(csv_file)
            df = df[df['BookId'] == bookId]
            if subfields:
                for subname in subfields:
                    metrics[name][subname]= df[subname]
            else:
                metrics[name] = df[name].iloc[0]
        except: 
            print('Cannot parse CSV')
        
def fill_in_figures(document, selected_imgs,fig_idx,image_type):
    barrier = TexCmd('FloatBarrier')
    document.document.append(barrier)
    subsec = TexCmd('subsection', args=[BraceGroup(f'{image_type}')]) # Add sections if desired, use \FloatBarrier from {placeins}
    document.document.append(subsec)
    for image_file in selected_imgs:
        # image_path = os.path.join(figures_paths, image_file).replace("\\", "/")
        #img_name = image_file.split('/')[-1].split('.')[0]
        #img_name = image_file.split('/')[-1].split('.')[:-1]
        #img_name_str = ""
        #for s in img_name:
        #    img_name_str += s
        image_name_full = os.path.basename(image_file)
        img_name, _ = os.path.splitext(image_name_full)
        img_n_mod = img_name.replace('_','-') # underscores in tex is bad
        fig_idx += 1
        # The tex-file is in the same location as images so img_name suffices. 
        # Latexmk or pdflatex should be called from that directory as well
        inc_graph = rf'\includegraphics[width=\columnwidth]{{{img_name}}}'
        # \subsection*{{ {img_name} }}
        image_section = TexSoup(rf'''
        \begin{{figure}}[!htpb]
        \centering
        	{inc_graph}
            \caption{{{img_n_mod}}}
	        \label{{fig:{fig_idx}}}
        \end{{figure}}
        ''')
        document.document.append(image_section)
        # document.document.append(TexCmd('pagebreak')) # Too forceful, better \usepackage[section]{placeins} an
    return fig_idx

# Example usage
def extract_parameters(xml_root):
    simul_params = {}
    for child in xml_root.find("Agents"):
        if child.tag == 'InitializationAgent':
            simul_params = add_params_to_dict(simul_params, child, 'initAgent')
        if child.tag == 'StylizedTraderAgent':
            if int(child.attrib['instanceCount']) > 0:
                if (float(child.attrib['sigmaF'])!= 0 and float(child.attrib['sigmaC'])!= 0) or (float(child.attrib['sigmaN'])!= 0 and float(child.attrib['sigmaC'])!= 0) or (float(child.attrib['sigmaF'])!= 0 and float(child.attrib['sigmaN'])!= 0) or (float(child.attrib['sigmaF'])!= 0 and float(child.attrib['sigmaN'])!= 0 and float(child.attrib['sigmaC'])!= 0): 
                    simul_params = add_params_to_dict(simul_params,child,'styAgent') #['sCount'] += int(child.attrib['instanceCount'])
                elif float(child.attrib['sigmaF']) != 0:
                    simul_params = add_params_to_dict(simul_params,child,'fundAgent') #simul_params['fCount'] += int(child.attrib['instanceCount'])
                elif float(child.attrib['sigmaC']) != 0:
                    simul_params = add_params_to_dict(simul_params,child,'chartAgent') # simul_params['cCount'] += int(child.attrib['instanceCount'])
                elif float(child.attrib['sigmaN']) != 0:
                    simul_params = add_params_to_dict(simul_params,child,'noiseAgent') # simul_params['nCount'] += int(child.attrib['instanceCount'])
        if child.tag == 'HighFrequencyTraderAgent':
            if int(child.attrib['instanceCount']) > 0:
                simul_params = add_params_to_dict(simul_params,child,'hftAgent') # simul_params['nCount'] += int(child.attrib['instanceCount'])
    return simul_params

def add_params_to_dict(simul_params, child, name):
    for key,value in child.attrib.items():
        # TODO remove unnecessary key pairs
        if 'debug' in key: 
            continue
        if 'name' in key:
            simul_params[name] = f'{name}'
        else:
            simul_params[name + key.replace('_','-')] = value.replace('_','-')
    return simul_params
def compile_latex(file_path):
    '''
    Call latexmk to generate the pdf from tex file using pdflatex. 
    Only minimal output is given pack
    Requirements latexmk and pdflatex
    Tested only with texlive-full package
    '''
    print("Initiating latexmk with pdflatex and minimal output...\n")
    pwd = os.getcwd()
    folder,filename = os.path.split(file_path)
    # this can be removed if we put output dir to the latexmk call
    os.chdir(folder)
    #Clean if exists
    subprocess.run(
        ["latexmk","-c",file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    #  Compile
    result = subprocess.run(
        ["latexmk", "-pdf", "-pdflatex=pdflatex -interaction=nonstopmode", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    output = result.stdout
    errors = result.stderr

    if output:
        print("LaTeXmk Output:")
        for line in output.split('\n'):
            if re.match(r'^.*:[0-9]*: .*$', line):
                print(line)

    if errors:
        print("\nLaTeXmk Errors:")
        print(errors)
    os.chdir(pwd)
    print("\nLaTeX compilation finished.\n")
if __name__ == "__main__":
    # This file can be used separately
    template = 'rayleigh-template.tex' # Add folder path later
    dir = os.getcwd()
    search_pattern = os.path.join(dir, 'logs/*')
    dirs = [p for p in glob(search_pattern) if os.path.isdir(p)]
    if not dirs:
        print("No directories found matching the pattern.")
    dirs.sort(key=os.path.getmtime, reverse=True)    
    if len(sys.argv) == 1:
        latest_dir = dirs[0] if len(dirs) >0 else dir
    else:
        latest_dir = sys.argv[1]
    search_pattern = os.path.join(latest_dir, '*L2*')
    l2_files = sorted(glob(search_pattern))
    search_pattern = os.path.join(latest_dir, '*L3*')
    l3_files = sorted(glob(search_pattern))
    try:
        xml = ET.parse(os.path.join(latest_dir, 'config.xml')).getroot()
    except:
        xml = None
    for l2_file,l3_file in zip(l2_files,l3_files):
        bookId = int(l2_file.split('.')[-2].split('-')[-1])
        out_dir = os.path.join(latest_dir,f"book_{bookId}") # Save directory in StylizedTraderReporting
        figures_paths = sorted(glob(out_dir + '/*.png'))  #"path_to_your_image_folder"  # Replace with your folder path
        tex_name = f'rayleigh-sim-report-{bookId}'# Replace with desired output .tex file name
        output_tex_file = os.path.join(out_dir,f'{tex_name}.tex')  
    
        generate_tex_with_texsoup(template,figures_paths,xml,latest_dir,output_tex_file)
        compile_latex(output_tex_file)
DEFAULT_SOUP = TexSoup(r"""
        \documentclass[twocolumn, 9pt]{extarticle}
        \usepackage[utf8]{inputenc}
        \usepackage[T1]{fontenc}
        \usepackage{graphicx}
        \usepackage{tabularx}
        \usepackage{authblk}
        \usepackage[section]{placeins}
        \usepackage[a4paper, margin=1in]{geometry}
        \title{TITLE}
        %%% CHANGE HERE
        \author{Add your name}
        \affil[1]{Rayleigh Research Oy}
        \begin{document}
        \maketitle
        Confidential
        \section{PARAMS}
        \begin{table}[htpb]
	    \centering
	    \begin{tabularx}{\columnwidth}{|c|X|}
		\hline
		\textbf{{Param}} & \textbf{{Value}}\\ \hline
	    \end{tabularx}
	    \caption{Parameters}
	    \label{tab:params}
        \end{table}
        \section{Figures}
        \end{document}
        """)

