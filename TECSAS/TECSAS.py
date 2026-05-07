# Copyright (c) 2020-2023 The Center for Theoretical Biological Physics (CTBP) - Rice University
# This file is from the TECSAS project, released under the MIT License. 

import os, glob, requests, shutil, urllib, gzip, math
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from urllib.parse import urlparse
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F


class data_process:
    def __init__(self, cell_line='GM12878', assembly='hg19',organism='human',signal_type='signal p-value',file_format='bigWig',
                ref_cell_line_path='tmp_meta',ref_chrm_size=None,num_sub=None,cell_line_path=None,types_path=None,
                histones=True,tf=False,atac=False,small_rna=False,total_rna=False,n_states=19,
                extra_filter='',res=50,chromosome_sizes=None,require_ENCODE=False):
        import pyBigWig as pybw
        self.pybw = pybw
        pt = os.path.dirname(os.path.realpath(__file__))
        self.path_to_share = os.path.join(pt,'share/')
        self.cell_line=cell_line
        self.assembly=assembly
        self.signal_type=signal_type
        if cell_line_path==None:
            self.cell_line_path=cell_line+'_'+assembly
        else:
            self.cell_line_path=cell_line_path
        self.ref_cell_line='GM12878'
        self.ref_assembly='hg19'
        self.ref_cell_line_path=ref_cell_line_path
        if types_path!=None:
            self.types_path=types_path
        else:
            self.types_path=self.path_to_share+'/subcom_GM12878_50kb'
        self.hist=histones
        self.tf=tf
        self.atac=atac
        self.small_rna=small_rna
        self.total_rna=total_rna
        self.n_states=n_states
        self.extra_filter=extra_filter
        self.res=res
        self.organism=organism.lower()
        if file_format.lower()=='bigwig':self.file_format='bigWig'
        elif file_format.lower()=='bed': self.file_format='bed+narrowPeak'
        #Default to empty experiment lists; populated below only when require_ENCODE=True
        self.experiments_unique=np.array([])
        self.es_unique=[]
        #Define translation dictinaries between aminoacids, intensity of Chip-seq signal/RNASeq count and states of the model
        self.RES_TO_INT = {
                'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
                'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
                'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
                'S': 16, 'T': 17, 'V': 18, 'W':19, 'Y':20,
                '-':21, '.':21, '~':21,}
        self.INT_TO_RES = {self.RES_TO_INT[k]:k for k in self.RES_TO_INT.keys()}
        self.TYPE_TO_INT = {'A1':0,'A2':1,'B1':2,'B2':3,'B3':4,'B4':5,'NA':-1}
        self.INT_TO_TYPE = {self.TYPE_TO_INT[k]:k for k in self.TYPE_TO_INT.keys()}
        # Define assembly of the target cell
        if assembly=='GRCh38':
            self.chrm_size = np.array([4980,4844,3966,3805,3631,3417,3187,2903,2768,2676,2702,2666,2288,2141,2040,1807,1666,1608,1173,1289,935,1017,3121])*50/self.res
            self.chrom_l={'chr1':248956422,'chr2':242193529,'chr3':198295559,'chr4':190214555,'chr5':181538259,'chr6':170805979,'chr7':159345973,'chrX':156040895,'chr8':145138636,'chr9':138394717,'chr11':135086622,'chr10':133797422,'chr12':133275309,'chr13':114364328,'chr14':107043718,'chr15':101991189,'chr16':90338345,'chr17':83257441,'chr18':80373285,'chr20':64444167,'chr19':58617616,'chrY':	57227415,'chr22':50818468,'chr21':46709983}
        elif assembly=='hg19':
            self.chrm_size = np.array([4990,4865,3964,3828,3620,3424,3184,2931,2826,2712,2703,2679,2307,2148,2052,1810,1626,1564,1184,1262,964,1028,3105])*50/self.res
            self.chrom_l={'chr1':249250621,'chr2':243199373,'chr3':198022430,'chr4':191154276,'chr5':180915260,'chr6':171115067,'chr7':159138663,'chrX':155270560,'chr8':146364022,'chr9':141213431,'chr10':135534747,'chr11':135006516,'chr12':133851895,'chr13':115169878,'chr14':107349540,'chr15':102531392,'chr16':90354753,'chr17':81195210,'chr18':78077248,'chr20':63025520,'chrY':59373566,'chr19':59128983,'chr22':51304566,'chr21':48129895}
        else: # If not GRCh38 or hg19 the chromosome sizes are required
            if chromosome_sizes == None: 
                raise ValueError("Need to specify chromosome sizes for assembly: {}".format(assembly))
            self.chrm_size = np.array(chromosome_sizes)/(self.res*1000)
            self.chrom_l = np.array(chromosome_sizes)
        self.chrm_size=np.round(self.chrm_size+0.1).astype(int)
        if ref_chrm_size == None:
            self.ref_chrm_size = np.array([4990,4865,3964,3828,3620,3424,3184,2931,2826,2712,2703,2679,2307,2148,2052,1810,1626,1564,1184,1262,964,1028,1028])*50/self.res
            self.ref_chrm_size = np.round(self.ref_chrm_size+0.1).astype(int)
            self.custom_chrom_size = False
        else:
            self.num_sub = num_sub
            self.TYPE_TO_INT = {'A1':0,'A2':1,'A3':2,'A4':3,'A5':4,'A6':5,'A7':6,'A8':7,'A9':8,'NA':-1}
            self.INT_TO_TYPE = {self.TYPE_TO_INT[k]:k for k in self.TYPE_TO_INT.keys()}
            print('Custom N of loci per chromosome: \n',ref_chrm_size)
            self.ref_chrm_size = ref_chrm_size
            self.custom_chrom_size = True
        if require_ENCODE==True:
            #Retrieves the available experiments on GM12878-hg19 to assess the download of experiments on the target cell
            #Prepare url to request information
            url='https://www.encodeproject.org/metadata/?type=Experiment&'
            if self.hist==True:
                url=url+'assay_title=Histone+ChIP-seq'
            if self.tf==True:
                url=url+'&assay_title=TF+ChIP-seq'
            if self.atac==True:
                url=url+'&assay_title=ATAC-seq'
            if self.small_rna==True:
                url=url+'&assay_title=small+RNA-seq'
            if self.total_rna==True:
                url=url+'&assay_title=total+RNA-seq'
            self.url_ref=url+'&biosample_ontology.term_name='+self.ref_cell_line+'&files.file_type='+self.file_format
            #Request information
            r = requests.get(self.url_ref)
            #Decode information requested
            content=str(r.content)
            experiments=[]
            for k in content.split('\\n')[:-1]:
                l=k.split('\\t')
                if l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='Histone ChIP-seq':
                    experiments.append(l[22])
                elif l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='ATAC-seq':
                    experiments.append(l[7])
                elif l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='TF ChIP-seq':
                    experiments.append(l[22])
                elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='small RNA-seq':
                    experiments.append('plus-small-RNA-seq')
                elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='total RNA-seq':
                    experiments.append('plus-total-RNA-seq')          
                elif l[5]==self.ref_assembly and l[4]=='minus strand signal of all reads' and l[7]=='small RNA-seq':
                    experiments.append('minus-small-RNA-seq')
                elif l[5]==self.ref_assembly and l[4]=='minus strand signal of all reads' and l[7]=='total RNA-seq':
                    experiments.append('minus-total-RNA-seq')          
            #Extract set of experiments found on GM12878-hg19 
            self.experiments_unique=np.unique(experiments)
            self.es_unique=[]   
            for e in self.experiments_unique:
                self.es_unique.append(e.split('-human')[0])

    def process_replica_bw(self,line,cell_line_path,chrm_size):
        R"""
        Preprocess function for each replica formated in bigwig file
        Args: 
            line (list, required):
                Information about the replica: name, ENCODE id and replica id
            cell_line_path (str, required):
                Path to target cell type data
            chrm_size (list, required):
                Chromosome sizes based on the assembly
        """
        #Extract experiment id
        text=line.split()[0]
        #Extract experiment type
        exp=line.split()[1]
        #Extract number id of experiment
        count=line.split()[2]
        #Extract accession number (ENCODE)
        sr_number=line.split()[3]
        #Experiment directory 
        exp_path=cell_line_path+'/'+exp+'_'+str(count)
        #Generate bookeeping files
        if 'human' in exp.split('-'): ext='human'
        else: ext=self.organism
        if exp.split('-'+ext)[0] in self.es_unique:
            try:
                os.mkdir(exp_path)
            except:
                print('Directory ',exp_path,' already exist')
            with open(exp_path+'/exp_name.txt', 'w') as f:
                f.write(text+' '+exp+'\n')
            with open(exp_path+'/exp_accession.txt', 'w') as f:
                f.write(sr_number+' '+exp+'\n')
                
            #Load data from server. ENCODE redirects file URLs to AWS S3
            #(307 -> encode-public.s3.amazonaws.com), and pyBigWig's libcurl
            #path does not follow that redirect reliably. Download to a temp
            #file with requests (which follows redirects) and open locally.
            import tempfile
            url = "https://www.encodeproject.org/files/"+text+"/@@download/"+text+".bigWig"
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".bigWig")
            os.close(tmp_fd)
            try:
                with requests.get(url, stream=True, allow_redirects=True, timeout=300) as resp:
                    resp.raise_for_status()
                    with open(tmp_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=1024*1024):
                            f.write(chunk)
                bw = self.pybw.open(tmp_path)
                #Process replica for numbered chromosomes
                for chr in range(1,len(chrm_size)):
                    signal = bw.stats("chr"+str(chr), type="mean", nBins=chrm_size[chr-1])
                    #Process signal and binning
                    signal=np.array(signal)
                    per=np.percentile(signal[signal!=None],95)
                    per_min=np.percentile(signal[signal!=None],5)
                    signal[signal==None]=per_min
                    signal[signal<per_min]=per_min
                    signal[signal>per]=per
                    signal=signal-per_min
                    signal=(signal-np.mean(signal))/np.std(signal)
                    #Save data for each chromosome
                    with open(exp_path+'/chr'+str(chr)+'.track', 'w') as f:
                        f.write("#chromosome file number of beads\n"+str(chrm_size[chr-1]))
                        f.write("#\n")
                        f.write("#bead, signal, discrete signal\n")
                        for i in range(len(signal)):
                            f.write(str(i)+" "+str(signal[i])+" "+str(signal[i])+"\n")
                #Process seperatly chromosome X
                chr='X'
                signal = bw.stats("chr"+chr, type="mean", nBins=chrm_size[-1])
                #Process signal and binning
                signal=np.array(signal)
                per=np.percentile(signal[signal!=None],95)
                per_min=np.percentile(signal[signal!=None],5)
                signal[signal==None]=per_min
                signal[signal<per_min]=per_min
                signal[signal>per]=per
                signal=signal-per_min
                signal=(signal-np.mean(signal))/np.std(signal)
                #Save data
                with open(exp_path+'/chr'+chr+'.track', 'w') as f:
                    f.write("#chromosome file number of beads\n"+str(chrm_size[-1]))
                    f.write("#\n")
                    f.write("#bead, signal, discrete signal\n")
                    for i in range(len(signal)):
                        f.write(str(i)+" "+str(signal[i])+" "+str(signal[i])+"\n")
                bw.close()
            finally:
                try: os.remove(tmp_path)
                except OSError: pass
            return exp

    def process_replica_bed(self,line,cell_line_path,chrm_size):
        R"""
        Preprocess function for each replica formated in bed files
        Args: 
            line (lsit, required):
                Information about the replica: name, ENCODE id and replica id
            cell_line_path (str, required):
                Path to target cell type data
            chrm_size (list, required):
                Chromosome sizes based on the assembly
        """
        #Extract experiment id
        text=line.split()[0]
        #Extract experiment type
        exp=line.split()[1]
        #Extract number id of experiment
        count=line.split()[2]
        #Extract accession number (ENCODE)
        sr_number=line.split()[3]
        
        #Experiment directory 
        exp_path=cell_line_path+'/'+exp+'_'+str(count)
        #Generate bookeeping files
        if 'human' in exp.split('-'): ext='human'
        else: ext=self.organism
        if exp.split('-'+ext)[0] in self.es_unique:
            try:
                os.mkdir(exp_path)
            except:
                print('Directory ',exp_path,' already exist')
            with open(exp_path+'/exp_name.txt', 'w') as f:
                f.write(text+' '+exp+'\n')
            with open(exp_path+'/exp_accession.txt', 'w') as f:
                f.write(sr_number+' '+exp+'\n')
            #Load data from server
            try:
                #Extract data from bed files
                exp_url="https://www.encodeproject.org/files/"+text+"/@@download/"+text+".bed.gz"
                response = urllib.request.urlopen(exp_url)
                gunzip_response = gzip.GzipFile(fileobj=response)
                content = gunzip_response.read()
                data=np.array([i.split('\t') for i in content.decode().split('\n')[:-1]])
                #Process replica for numbered chromosomes
                for chr in range(1,len(chrm_size)):
                    chrm_data=data[data[:,0]=='chr'+str(chr)][:,[1,2,6]].astype(float)
                    signal=np.zeros(chrm_size[chr-1])
                    ndx_small=np.floor(chrm_data[:,1]/(self.res*1000)) == np.floor(chrm_data[:,0]/(self.res*1000))
                    # Aggregate peak intensity
                    for ll in chrm_data[ndx_small]:
                        ndx=int(ll[0]/(self.res*1000))
                        if ndx<len(signal):
                            signal[ndx]+=ll[2]
                    for ll in chrm_data[~ndx_small]:
                        ndx1=int(ll[0]/(self.res*1000))
                        ndx2=int(ll[1]/(self.res*1000))
                        if ndx1<len(signal) and ndx2<len(signal):
                            p=(ndx2-ll[0]/(self.res*1000))/((ll[1]-ll[0])/(self.res*1000))
                            signal[ndx1]+=ll[2]*p
                            signal[ndx2]+=ll[2]*(1-p)
                    
                    per=np.percentile(signal[signal!=None],95)
                    per_min=np.percentile(signal[signal!=None],5)
                    signal[signal==None]=per_min
                    signal[signal<per_min]=per_min
                    signal[signal>per]=per
                    signal=signal-per_min
                    signal=signal*self.n_states/(per-per_min)
                    signal=np.round(signal.astype(float)).astype(int)
                    #Save data
                    with open(exp_path+'/chr'+str(chr)+'.track', 'w') as f:
                        
                        f.write("#chromosome file number of beads\n"+str(chrm_size[chr-1]))
                        f.write("#\n")
                        f.write("#bead, signal, discrete signal\n")
                        for i in range(len(signal)):
                            f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
                #Process seperatly chromosome X
                chr='X'
                chrm_data=data[data[:,0]=='chr'+chr][:,[1,2,6]].astype(float)
                signal=np.zeros(chrm_size[-1])
                ndx_small=np.floor(chrm_data[:,1]/(self.res*1000)) == np.floor(chrm_data[:,0]/(self.res*1000))
                # Aggregate peak intensity
                for ll in chrm_data[ndx_small]:
                    ndx=int(ll[0]/(self.res*1000))
                    if ndx<len(signal):
                        signal[ndx]+=ll[2]
                for ll in chrm_data[~ndx_small]:
                    ndx1=int(ll[0]/(self.res*1000))
                    ndx2=int(ll[1]/(self.res*1000))
                    if ndx1<len(signal) and ndx2<len(signal):
                        p=(ndx2-ll[0]/(self.res*1000))/((ll[1]-ll[0])/(self.res*1000))
                        signal[ndx1]+=ll[2]*p
                        signal[ndx2]+=ll[2]*(1-p)
                per=np.percentile(signal[signal!=None],95)
                per_min=np.percentile(signal[signal!=None],5)
                signal[signal==None]=per_min
                signal[signal<per_min]=per_min
                signal[signal>per]=per
                signal=signal-per_min
                signal=signal*self.n_states/(per-per_min)
                signal=np.round(signal.astype(float)).astype(int)
                #Save data
                with open(exp_path+'/chr'+chr+'.track', 'w') as f:
                    f.write("#chromosome file number of beads\n"+str(chrm_size[-1]))
                    f.write("#\n")
                    f.write("#bead, signal, discrete signal\n")
                    for i in range(len(signal)):
                        f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
                return exp
            
            except:
                print('This experiment was incomplete:',text,'\nit will not be used.')

    def download_and_process_cell_line_data(self,nproc=10,all_exp=True):
        R"""
        Download and preprocess target cell data for the D-nodes

        Args: 
            nproc (int, required):
                Number of processors dedicated to download and process data
            all_exp (bool, optional):
                Download and process all replicas for each experiment. Set as 'False' to download only 1 replica per experiment
        """
        #Create directory for target cell data
        try:
            os.mkdir(self.cell_line_path)
        except:
            print('Directory ',self.cell_line_path,' already exist')
            print('Deleting path and creating it anew')
            shutil.rmtree(self.cell_line_path)
            os.mkdir(self.cell_line_path)
        #Prepare url to fetch target cell data
        url='https://www.encodeproject.org/metadata/?type=Experiment&'+self.extra_filter
        if self.hist==True:
            url=url+'assay_title=Histone+ChIP-seq'
        if self.tf==True:
            url=url+'&assay_title=TF+ChIP-seq'
        if self.atac==True:
            url=url+'&assay_title=ATAC-seq'
        if self.small_rna==True:
            url=url+'&assay_title=small+RNA-seq'
        if self.total_rna==True:
            url=url+'&assay_title=total+RNA-seq'
        self.url_cell_line=url+'&biosample_ontology.term_name='+self.cell_line+'&files.file_type='+self.file_format
        # Request data from ENCODE server
        r = requests.get(self.url_cell_line)
        content=str(r.content)
        experiments=[]
        with open(self.cell_line_path+'/meta.txt', 'w') as f:
            for k in content.split('\\n')[:-1]:
                l=k.split('\\t')
                if l[5]==self.assembly and l[4]==self.signal_type and l[7]=='Histone ChIP-seq':
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.assembly and l[4]==self.signal_type and l[7]=='ATAC-seq':
                    f.write(l[0]+' '+l[7]+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.assembly and l[4]==self.signal_type and l[7]=='TF ChIP-seq':
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.assembly and l[4]=='plus strand signal of all reads' and l[7]=='small RNA-seq':
                    f.write(l[0]+' '+'plus-small-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.assembly and l[4]=='plus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'plus-total-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.assembly and l[4]=='minus strand signal of all reads' and l[7]=='small RNA-seq':
                    f.write(l[0]+' '+'minus-small-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.assembly and l[4]=='minus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'minus-total-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
        # Record experiments, number of replicas and their accession numbers
        count=0
        self.exp_found={}
        exp_name=''
        list_names=[]
        with open(self.cell_line_path+'/meta.txt') as fp:
            Lines = fp.readlines()
            for line in Lines:
                count += 1
                text=line.split()[0]
                exp=line.split()[1]
                sr_number=line.split()[-1]
                #Register if experiment is new
                if exp!=exp_name:
                    try:
                        count=self.exp_found[exp]+1
                    except:
                        count=1
                    exp_name=exp
                self.exp_found[exp]=count
                if all_exp==True:
                    list_names.append(text+' '+exp+' '+str(count)+' '+sr_number)
                else:
                    if count==1:
                        list_names.append(text+' '+exp+' '+str(count)+' '+sr_number)
        #Process signal for each replica in parallel 
        print('Number of replicas:', len(list_names))
        if self.file_format=='bigWig':
            self.successful_exp = Parallel(n_jobs=nproc)(delayed(self.process_replica_bw)(list_names[i],self.cell_line_path,self.chrm_size) 
                                        for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))
        else:
            self.successful_exp = Parallel(n_jobs=nproc)(delayed(self.process_replica_bed)(list_names[i],self.cell_line_path,self.chrm_size) 
                                        for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))
        #Record the replicas/experiments that were successfully processed
        self.successful_exp= [i for i in self.successful_exp if i]
        self.successful_unique_exp=np.unique(self.successful_exp)
        self.su_unique=[]   
        for e in self.successful_unique_exp:
            self.su_unique.append(e.split('-'+self.organism)[0])
        print('Experiments found in ENCODE for the selected cell line:')
        self.unique=[]
        #Save the set of unique experiments found in the target cell
        with open(self.cell_line_path+'/unique_exp.txt', 'w') as f:
            for e in self.experiments_unique:
                if e.split('-human')[0] in self.su_unique:
                    f.write(e.split('-human')[0]+'\n')
                    print(e.split('-human')[0])
                    self.unique.append(e)
        if len(self.unique) > 4:
            print('Predictions would use: ',len(self.unique),' experiments')
        else:
            print('This sample only has ',len(self.unique),' experiments. We do not recommend prediction on samples with less than 5 different experiments.')

    def download_and_process_ref_data(self,nproc,all_exp=True):
        R"""
        Download and preprocess reference data for the D-nodes

        Args: 
            nproc (int, required):
                Number of processors dedicated to download and process data
            all_exp (bool, optional):
                Download and process all replicas for each experiment. Set as 'False' to download only 1 replica per experiment
        """
        #Create directory for target cell data
        try:
            os.mkdir(self.ref_cell_line_path)
        except:
            print('Directory ',self.ref_cell_line_path,' already exist')
            print('Deleting path and creating it anew')
            shutil.rmtree(self.ref_cell_line_path)
            os.mkdir(self.ref_cell_line_path)
        #Prepare url to fetch target cell data
        url='https://www.encodeproject.org/metadata/?type=Experiment&'
        if self.hist==True:
            url=url+'assay_title=Histone+ChIP-seq'
        if self.tf==True:
            url=url+'&assay_title=TF+ChIP-seq'
        if self.atac==True:
            url=url+'&assay_title=ATAC-seq'
        if self.small_rna==True:
            url=url+'&assay_title=small+RNA-seq'
        if self.total_rna==True:
            url=url+'&assay_title=total+RNA-seq'
        self.url_ref=url+'&biosample_ontology.term_name='+self.ref_cell_line+'&files.file_type='+self.file_format
        # Record experiments, number of replicas and their accession numbers
        r = requests.get(self.url_ref)
        content=str(r.content)
        experiments=[]
        with open(self.ref_cell_line_path+'/meta.txt', 'w') as f:
            for k in content.split('\\n')[:-1]:
                l=k.split('\\t')
                if l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='Histone ChIP-seq':
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='ATAC-seq':
                    f.write(l[0]+' '+l[7]+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.ref_assembly and l[4]==self.signal_type and l[7]=='TF ChIP-seq':
                    f.write(l[0]+' '+l[22]+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='small RNA-seq':
                    f.write(l[0]+' '+'plus-small-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.ref_assembly and l[4]=='plus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'plus-total-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.ref_assembly and l[4]=='minus strand signal of all reads' and l[7]=='small RNA-seq':
                    f.write(l[0]+' '+'minus-small-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
                elif l[5]==self.ref_assembly and l[4]=='minus strand signal of all reads' and l[7]=='total RNA-seq':
                    f.write(l[0]+' '+'minus-total-RNA-seq'+' '+l[5]+' '+l[4]+' '+l[6]+'\n')
        # Record experiments, number of replicas and their accession numbers
        ref_chrm_size = self.ref_chrm_size
        count=0
        exp_found={}
        exp_name=''
        list_names=[]
        with open(self.ref_cell_line_path+'/meta.txt') as fp:
            Lines = fp.readlines()
            for line in Lines:
                count += 1
                text=line.split()[0]
                exp=line.split()[1]
                sr_number=line.split()[-1]
                #Register if experiment is new
                if (exp.split('-human')[0] in self.su_unique) or (text.split('-human')[0] in self.su_unique):
                    if exp!=exp_name:
                        try:
                            count=exp_found[exp]+1
                        except:
                            count=1
                        exp_name=exp
                    exp_found[exp]=count
                    if all_exp==True:
                        list_names.append(text+' '+exp+' '+str(count)+' '+sr_number)
                    else:
                        if count==1:
                            list_names.append(text+' '+exp+' '+str(count)+' '+sr_number)
        print('Number of replicas:', len(list_names))
        #Process signal for each replica in parallel 
        if self.file_format=='bigWig':
            results = Parallel(n_jobs=nproc)(delayed(self.process_replica_bw)(list_names[i],self.ref_cell_line_path,ref_chrm_size) 
                                    for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))
        else:
            results = Parallel(n_jobs=nproc)(delayed(self.process_replica_bed)(list_names[i],self.ref_cell_line_path,ref_chrm_size) 
                                    for i in tqdm(range(len(list_names)), desc="Process replicas",bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'))
        print('Prediction will use:')
        with open(self.ref_cell_line_path+'/unique_exp.txt', 'w') as f:
            for e in self.unique:
                if e in exp_found.keys():
                    f.write(e.split('-'+self.organism)[0]+'\n')
                    print(e.split('-'+self.organism)[0])

    def custom_bw_track(self,experiment,bw_file):
        R"""
        Function to introduce custom bigwig tracks
        
        Args: 
            experiment (str, required):
                Name of the experiment
            bw_file (str, required):
                Path to the custom track
        """
        #Format the name of the experiment
        if not self.organism in experiment: experiment=experiment+'-'+self.organism
        if not experiment.split('-'+self.organism)[0] in self.es_unique: 
            print('This experiment is not found in the training set, then cannot be used.')
            return 0
        if experiment in self.exp_found.keys():
            print('This target has replicas already')
            print('The new track will be addded as a different replica of the same target')
            #Experiment directory
            count=self.exp_found[experiment]+1           
        else:
            print('This target has no replicas')
            print('The new track will be added a the first replica of the target')
            #Experiment directory
            count=1
        exp_path=self.cell_line_path+'/'+experiment+'_'+str(count)
        print(exp_path)
        # Create directory for new replica
        try:
            os.mkdir(exp_path)
        except:
            print('Directory ',exp_path,' already exist')
        with open(exp_path+'/exp_name.txt', 'w') as f:
            f.write(experiment+' '+experiment+'\n')
        #Load data from track
        try:
            bw = self.pybw.open(bw_file)
            for chr in range(1,len(self.chrm_size)):
                signal = bw.stats("chr"+str(chr), type="mean", nBins=self.chrm_size[chr-1])
                #Process signal and binning
                signal=np.array(signal)
                per=np.percentile(signal[signal!=None],95)
                per_min=np.percentile(signal[signal!=None],5)
                signal[signal==None]=per_min
                signal[signal<per_min]=per_min
                signal[signal>per]=per
                signal=signal-per_min
                signal=signal*self.n_states/(per-per_min)
                signal=np.round(signal.astype(float)).astype(int)
                #Save data
                with open(exp_path+'/chr'+str(chr)+'.track', 'w') as f:
                    f.write("#chromosome file number of beads\n"+str(self.chrm_size[chr-1]))
                    f.write("#\n")
                    f.write("#bead, signal, discrete signal\n")
                    for i in range(len(signal)):
                        f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
            chr='X'
            signal = bw.stats("chr"+chr, type="mean", nBins=self.chrm_size[-1])
            #Process signal and binning
            signal=np.array(signal)
            per=np.percentile(signal[signal!=None],95)
            per_min=np.percentile(signal[signal!=None],5)
            signal[signal==None]=per_min
            signal[signal<per_min]=per_min
            signal[signal>per]=per
            signal=signal-per_min
            signal=signal*self.n_states/(per-per_min)
            signal=np.round(signal.astype(float)).astype(int)
            #Save data
            with open(exp_path+'/chr'+chr+'.track', 'w') as f:
                f.write("#chromosome file number of beads\n"+str(self.chrm_size[-1]))
                f.write("#\n")
                f.write("#bead, signal, discrete signal\n")
                for i in range(len(signal)):
                    f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
            # Register new replica in the existing set of experiments
            if experiment in self.exp_found.keys():
                self.exp_found[experiment]=self.exp_found[experiment]+1
            else:
                self.exp_found[experiment]=1
                self.successful_unique_exp=np.append(self.successful_unique_exp,experiment)
                self.su_unique.append(experiment.split('-'+self.organism)[0])
                with open(self.cell_line_path+'/unique_exp.txt', 'a') as f:
                    f.write(experiment.split('-'+self.organism)[0]+'\n')
                    self.unique.append(experiment)
            return experiment
        except:
            print('This experiment was incomplete:',experiment,'\nit will not be used.')

    def custom_bed_track(self,experiment,bed_file):
        R"""
        Function to introduce custom bed tracks
        
        Args: 
            experiment (str, required):
                Name of the experiment
            bed_file (str, required):
                Path to the custom track
        """
        #Format the name of the experiment
        if not self.organism in experiment: experiment=experiment+'-'+self.organism
        if not experiment.split('-'+self.organism)[0] in self.es_unique: 
            print('This experiment is not found in the training set, then cannot be used.')
            return 0
        if experiment in self.exp_found.keys():
            print('This target has replicas already')
            print('The new track will be addded as a different replica of the same target')
            #Experiment directory
            count=self.exp_found[experiment]+1           
        else:
            print('This target has no replicas')
            print('The new track will be added a the first replica of the target')
            #Experiment directory
            count=1
        exp_path=self.cell_line_path+'/'+experiment+'_'+str(count)
        print(exp_path)
        # Create directory for new replica
        try:
            os.mkdir(exp_path)
        except:
            print('Directory ',exp_path,' already exist')
        with open(exp_path+'/exp_name.txt', 'w') as f:
            f.write(experiment+' '+experiment+'\n')
        def get_records(bed_file):
            try:
                response = urllib.request.urlopen(bed_file)
                gunzip_response = gzip.GzipFile(fileobj=response)
                content = gunzip_response.read()
                data=np.array([i.split('\t') for i in content.decode().split('\n')[:-1]])
            except:
                try:
                    gunzip_response = gzip.GzipFile(bed_file)
                    content = gunzip_response.read()
                    data=np.array([i.split('\t') for i in content.decode().split('\n')[:-1]])
                except:
                    data=np.loadtxt(bed_file,dtype=str)
            return data
        #Load data from track
        try:
            data=get_records(bed_file)
            get_records(bed_file)
            for chr in range(1,len(self.chrm_size)):
                chrm_data=data[data[:,0]=='chr'+str(chr)][:,[1,2,6]].astype(float)
                signal=np.zeros(self.chrm_size[chr-1])
                ndx_small=np.floor(chrm_data[:,1]/(self.res*1000)) == np.floor(chrm_data[:,0]/(self.res*1000))
                # Aggregate peak intensity
                for ll in chrm_data[ndx_small]:
                    ndx=int(ll[0]/(self.res*1000))
                    if ndx<len(signal):
                        signal[ndx]+=ll[2]
                for ll in chrm_data[~ndx_small]:
                    ndx1=int(ll[0]/(self.res*1000))
                    ndx2=int(ll[1]/(self.res*1000))
                    if ndx1<len(signal) and ndx2<len(signal):
                        p=(ndx2-ll[0]/(self.res*1000))/((ll[1]-ll[0])/(self.res*1000))
                        signal[ndx1]+=ll[2]*p
                        signal[ndx2]+=ll[2]*(1-p)
                #Process signal and binning
                signal=np.array(signal)
                per=np.percentile(signal[signal!=None],95)
                per_min=np.percentile(signal[signal!=None],5)
                signal[signal==None]=per_min
                signal[signal<per_min]=per_min
                signal[signal>per]=per
                signal=signal-per_min
                signal=signal*self.n_states/(per-per_min)
                signal=np.round(signal.astype(float)).astype(int)
                #Save data
                with open(exp_path+'/chr'+str(chr)+'.track', 'w') as f:
                    f.write("#chromosome file number of beads\n"+str(self.chrm_size[chr-1]))
                    f.write("#\n")
                    f.write("#bead, signal, discrete signal\n")
                    for i in range(len(signal)):
                        f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
            chr='X'
            chrm_data=data[data[:,0]=='chr'+chr][:,[1,2,6]].astype(float)
            signal=np.zeros(self.chrm_size[-1])
            ndx_small=np.floor(chrm_data[:,1]/(self.res*1000)) == np.floor(chrm_data[:,0]/(self.res*1000))
            # Aggregate peak intensity
            for ll in chrm_data[ndx_small]:
                ndx=int(ll[0]/(self.res*1000))
                if ndx<len(signal):
                    signal[ndx]+=ll[2]
            for ll in chrm_data[~ndx_small]:
                ndx1=int(ll[0]/(self.res*1000))
                ndx2=int(ll[1]/(self.res*1000))
                if ndx1<len(signal) and ndx2<len(signal):
                    p=(ndx2-ll[0]/(self.res*1000))/((ll[1]-ll[0])/(self.res*1000))
                    signal[ndx1]+=ll[2]*p
                    signal[ndx2]+=ll[2]*(1-p)
            #Process signal and binning
            signal=np.array(signal)
            per=np.percentile(signal[signal!=None],95)
            per_min=np.percentile(signal[signal!=None],5)
            signal[signal==None]=per_min
            signal[signal<per_min]=per_min
            signal[signal>per]=per
            signal=signal-per_min
            signal=signal*self.n_states/(per-per_min)
            signal=np.round(signal.astype(float)).astype(int)
            #Save data
            with open(exp_path+'/chr'+chr+'.track', 'w') as f:
                f.write("#chromosome file number of beads\n"+str(self.chrm_size[-1]))
                f.write("#\n")
                f.write("#bead, signal, discrete signal\n")
                for i in range(len(signal)):
                    f.write(str(i)+" "+str(signal[i])+" "+str(signal[i].astype(int))+"\n")
            if experiment in self.exp_found.keys():
                self.exp_found[experiment]=self.exp_found[experiment]+1
            else:
                self.exp_found[experiment]=1
                self.successful_unique_exp=np.append(self.successful_unique_exp,experiment)
                self.su_unique.append(experiment.split('-'+self.organism)[0])
                with open(self.cell_line_path+'/unique_exp.txt', 'a') as f:
                    f.write(experiment.split('-'+self.organism)[0]+'\n')
                    self.unique.append(experiment)
            return experiment
        except:
            print('This experiment was incomplete:',experiment,'\nit will not be used.')

    def build_state_vector(self,int_types,all_averages):
        R"""
        Builds the set of state vectors used on the training process

        Args: 
            int_types (list, required):
                Genomic annotations
            all_averages (list, required):
                D-node data 
        """
        #Aggregate tracks by with data from other loci l-2, l-1, l, l+1, l+2
        #l+1
        shift1=np.copy(all_averages)
        shift1[:,:-1]=all_averages[:,1:]
        shift1[:,-1]=np.zeros(len(shift1[:,-1]))
        #l+2
        shift2=np.copy(all_averages)
        shift2[:,:-1]=shift1[:,1:]
        shift2[:,-1]=np.zeros(len(shift1[:,-1]))
        #l-1
        shift_1=np.copy(all_averages)
        shift_1[:,1:]=all_averages[:,:-1]
        shift_1[:,0]=np.zeros(len(shift_1[:,-1]))
        #l-2
        shift_2=np.copy(all_averages)
        shift_2[:,1:]=shift_1[:,:-1]
        shift_2[:,0]=np.zeros(len(shift1[:,-1]))
        #Stack shifted tracks and subtypes labels
        all_averages=np.vstack((int_types,shift_2,shift_1,all_averages,shift1,shift2))
        #To train, we exclude the NA and B4 subcompartments
        if self.custom_chrom_size == False:
            ndx=(all_averages[0,:]!=-1) * (all_averages[0,:]!=5)
        else:
            ndx=(all_averages[0,:]!=-1)
        all_averages=all_averages[:,ndx]
        all_averages=all_averages
        return all_averages

    def get_tmatrix(self,chrms,silent=False,unique_file=None):
        R"""
        Extract the training data

        Args: 
            chrms (list, optional):
                Set of chromosomes from the reference data used as the training set
            silent (bool, optional):
                Silence outputs
        """
        #Load types from Rao et al 2014 paper
        types=[]
        for chr in chrms:
            if self.res==50 and self.custom_chrom_size==False:
                types.append(np.loadtxt(self.types_path+'/chr'+str(chr)+'_beads.txt.original',delimiter=' ',dtype=str)[:,1])
            else:
                tmp=list(np.loadtxt(self.types_path+'/chr'+str(chr)+'_beads.txt.original',delimiter=' ',dtype=str)[:,1])
                if len(tmp) < self.ref_chrm_size[chr-1]:
                    diff=self.ref_chrm_size[chr-1] - len(tmp)
                    for i in range( diff ):
                        tmp.append('NA')
                types.append(tmp)
        types=np.concatenate(types)
        int_types=np.array(list(map(self.TYPE_TO_INT.get, types)))
        #Check which experiments are available to train 
        if unique_file==None:
            unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str)
        else:
            unique=np.loadtxt(unique_file,dtype=str)
        if unique.shape==(): unique=[unique]
        if silent==False:print('To train the following experiments are used:')
        #Load each track and average over replicas
        all_averages=[]
        for u in unique:
            reps=[]
            if silent==False:print(u)
            for i in glob.glob(self.ref_cell_line_path+'/'+str(u)+'*'):
                tmp=[]
                try:
                    for chr in chrms:
                        _tmp=np.loadtxt(i+'/chr'+str(chr)+'.track',skiprows=3)[:,2]
                        tmp.append(_tmp)
                    tmp=np.concatenate(tmp)
                    reps.append(tmp)
                except:
                    if silent==False:print(i,' failed with at least one chromosome')
            reps=np.array(reps)
            ave_reps=np.mean(reps,axis=0)
            all_averages.append(ave_reps)
        #Build state vectors of the Potts model for training
        all_averages=np.array(all_averages)
        all_averages=self.build_state_vector(int_types,all_averages)
        return all_averages

    def filter_exp(self):
        R"""
        Performs assestment on experiment signal-to-noise ration based on mean and std of the signal compared to the GM12878 equivalent using chromosomes 1 and 2
        """
        a=[]
        for i in range(1,3):
            a.append(self.test_set(chr=i,silent=True))
        a=np.concatenate(a,axis=1)

        locus=2
        good_exp=0
        gexp=[]
        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str)
        #Compare mean and std of signal of target cell and GM12878 
        for exper in range(len(unique)):
            i=exper+len(unique)*locus
            if (np.abs(np.mean(a[i])-np.mean(self.tmatrix[i+1]))<1) and (np.std(a[i])-np.std(self.tmatrix[i+1])<2):
                good_exp=good_exp+1
                gexp.append(unique[exper]+'\n')
            else:
                print('Not using '+unique[exper],' to predict')
        #Records the experiments whose signal-to-noise ration is similar to GM12878
        with open(self.cell_line_path+'/unique_exp_filtered.txt','w') as f:    
            for i in gexp:
                f.write(i)
            print('Number of suitable experiments for prediction:',good_exp)
        if good_exp>0:
            os.system('mv '+self.cell_line_path+'/unique_exp.txt '+self.cell_line_path+'/unique_exp_bu.txt')
            os.system('mv '+self.cell_line_path+'/unique_exp_filtered.txt '+self.cell_line_path+'/unique_exp.txt')
        else:
            print('There are no experiment suitable for the prediction')

    def test_set(self,chr=1,silent=False):
        R"""
        Predicts and outputs the genomic annotations for chromosome X
        
        Args: 
            chr (int, required):
                Chromosome to extract input data fro the D-nodes
            silent (bool, optional):
                Avoid printing information 
        Returns:
            array (size of chromosome,5*number of unique experiments)
                D-node input data
        """
        if silent==False:print('Test set for chromosome: ',chr)        
        if chr!='X':
            types=["A1" for i in range(self.chrm_size[chr-1])]
        else:
            types=["A1" for i in range(self.chrm_size[-1])]
        int_types=np.array(list(map(self.TYPE_TO_INT.get, types)))
        unique=np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str) 
        if unique.shape==(): unique=[unique]
        #Load each track and average over 
        all_averages=[]
        for u in unique:
            reps=[]
            for i in glob.glob(self.cell_line_path+'/'+str(u)+'*'):
                tmp=[]
                try:
                    tmp=np.loadtxt(i+'/chr'+str(chr)+'.track',skiprows=3)[:,2]
                    reps.append(tmp)
                except:
                    if silent==False:print(i,' failed with at least one chromosome')
            reps=np.array(reps)
            ave_reps=np.mean(reps,axis=0)
            all_averages.append(ave_reps)
        all_averages=np.array(all_averages)
        chr_averages=self.build_state_vector(int_types,all_averages)
        return chr_averages[1:]

    def test_data(self,n_neigbors=2,n_predict=1,unique_file=None):
        tmp_all_matrix=self.get_tmatrix(range(1,23),silent=False,unique_file=unique_file)
        if unique_file==None:
            nfeatures=len(np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str))
        else:
            nfeatures=len(np.loadtxt(unique_file,dtype=str))
        #Populate data with neighbor information
        tmp=[]
        for l in range(np.max([n_neigbors,n_predict]),len(tmp_all_matrix[0])-np.max([n_neigbors,n_predict])):
            tmp.append(np.insert(np.concatenate(tmp_all_matrix[nfeatures*2+1:nfeatures*3+1,l-n_neigbors:l+n_neigbors+1].T),0,tmp_all_matrix[0,l-n_predict+1:l+n_predict]))
        all_matrix=np.array(tmp).T

        testmatrix=all_matrix
        test_set=testmatrix.T

        return test_set, all_matrix

    def training_data(self,n_neigbors=2,train_per=0.8,n_predict=1):
        tmp_all_matrix=self.get_tmatrix(range(1,23))
        nfeatures=len(np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str))
        #Populate data with neighbor information
        tmp=[]
        for l in range(np.max([n_neigbors,n_predict]),len(tmp_all_matrix[0])-np.max([n_neigbors,n_predict])):
            tmp.append(np.insert(np.concatenate(tmp_all_matrix[nfeatures*2+1:nfeatures*3+1,l-n_neigbors:l+n_neigbors+1].T),0,tmp_all_matrix[0,l-n_predict+1:l+n_predict]))
        all_matrix=np.array(tmp).T
        #Segment data between train, test and valiation sets
        tidx=np.random.choice(np.linspace(0,len(all_matrix[0])-1,len(all_matrix[0])).astype(int),size=int(train_per*len(all_matrix[0])),replace=False)
        ttidx=np.zeros(len(all_matrix[0])).astype(bool)
        ttidx[tidx]=1
        #Split genome data between train, validation and test set
        tmatrix=all_matrix[:,ttidx]
        vmatrix=all_matrix[:,~ttidx][:,::2]
        testmatrix=all_matrix[:,~ttidx][:,1::2]

        train_set=tmatrix.T
        validation_set=vmatrix.T
        test_set=testmatrix.T

        return train_set, validation_set, test_set, all_matrix, ttidx

    def training_data_chrom(self,n_neigbors=2,train_per=0.8,train_chroms=[1,2,3,4],n_predict=1):
        tmp_all_matrix=self.get_tmatrix(train_chroms)
        nfeatures=len(np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str))
        #Populate data with neighbor information
        tmp=[]
        for l in range(np.max([n_neigbors,n_predict]),len(tmp_all_matrix[0])-np.max([n_neigbors,n_predict])):
            tmp.append(np.insert(np.concatenate(tmp_all_matrix[nfeatures*2+1:nfeatures*3+1,l-n_neigbors:l+n_neigbors+1].T),0,tmp_all_matrix[0,l-n_predict+1:l+n_predict]))
        all_matrix=np.array(tmp).T
        #Segment data between train, test and valiation sets
        ttidx=np.ones(len(all_matrix[0])).astype(bool)
        #Store odd chromosomes as training set
        tmatrix=all_matrix
        all_matrix_train = all_matrix
        test_val_chroms = [i for i in range(1,23,1) if not i in train_chroms]
        tmp_all_matrix=self.get_tmatrix(test_val_chroms)
        nfeatures=len(np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str))
        #Populate data with neighbor information
        tmp=[]
        for l in range(np.max([n_neigbors,n_predict]),len(tmp_all_matrix[0])-np.max([n_neigbors,n_predict])):
            tmp.append(np.insert(np.concatenate(tmp_all_matrix[nfeatures*2+1:nfeatures*3+1,l-n_neigbors:l+n_neigbors+1].T),0,tmp_all_matrix[0,l-n_predict+1:l+n_predict]))
        all_matrix=np.array(tmp).T
        #Segment data between train, test and valiation sets
        tidx=np.random.choice(np.linspace(0,len(all_matrix[0])-1,len(all_matrix[0])).astype(int),size=int(train_per*len(all_matrix[0])),replace=False)
        #Store even chromosomes as training set
        vmatrix=all_matrix[:,::2]
        testmatrix=all_matrix[:,1::2]
        all_matrix_val_test = all_matrix
        train_set=tmatrix.T
        validation_set=vmatrix.T
        test_set=testmatrix.T
        ttidx=np.concatenate([ttidx,np.zeros(len(all_matrix[0])).astype(bool)])

        all_data=np.concatenate([all_matrix_train.T,all_matrix_val_test.T]).T
        return train_set, validation_set, test_set, all_data, ttidx


    def get_test_set(self,cellname,n_neigbors=2,n_predict=1,chrms=range(1,23)):
        #Initialize PyMEGABASE
        self.cell_line_path=cellname+'_GRCh38_zscore'
        print('looking for data in:',self.cell_line_path)
        test_cell=[]
        for chr in chrms:
            test_cell.append(self.test_set(chr=chr,silent=True))
        test_cell=np.concatenate(test_cell,axis=1)
        #Populate prediction set
        tmp=[]
        nfeatures=len(np.loadtxt(self.cell_line_path+'/unique_exp.txt',dtype=str))
        for l in range(n_neigbors,len(test_cell[0])-n_neigbors):
            tmp.append(np.insert(np.concatenate(test_cell[nfeatures*2:nfeatures*3,l-n_neigbors:l+n_neigbors+1].T),0,np.zeros(n_predict)))
        testmatrix=np.array(tmp).T
        nfeatures=(2*n_neigbors+1)*nfeatures
        test_set=testmatrix.T
        test_data=torch.tensor(test_set.astype(float))
        return test_data

class PositionalEncoding(nn.Module):
 
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
 
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
 
    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)
 
class TECSAS(nn.Module):
    def __init__(self, n_pred_loci: int, d_model: int, nhead: int, d_hid: int,
                nlayers: int, features: int, ostates: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.encoder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.fl = nn.Flatten()
        self.l2 = nn.Linear(features*d_model,ostates*(2*(n_pred_loci-1)+1))
        self.unfl = nn.Unflatten(-1,((2*(n_pred_loci-1)+1),ostates))
        self.init_weights()
 
    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.l2.bias.data.zero_()
        self.l2.weight.data.uniform_(-initrange, initrange)
 
    def partial_forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return(src)
 
    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output_tf = self.transformer_encoder(src, src_mask)
        output = self.fl(output_tf)
        output = self.l2(output)
        output = self.unfl(output)
        return  F.log_softmax(output, dim=-1), output_tf

class TECSAS_discrete(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                nlayers: int, features: int, ostates: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.encoder = nn.Embedding(ntoken, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.fl = nn.Flatten()
        self.l2 = nn.Linear(features*d_model,ostates)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.l2.bias.data.zero_()
        self.l2.weight.data.uniform_(-initrange, initrange)

    def partial_forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return(src)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output_tf = self.transformer_encoder(src, src_mask)
        output = self.fl(output_tf)
        output = self.l2(output)
        return  F.log_softmax(output, dim=-1), output_tf

def extract_selfattention_maps(model,data):
    with torch.no_grad():
        src=model.partial_forward(data,None)
        pred=model(data,None)[0].argmax(dim=1).cpu()

    attention_maps = []
    num_layers = model.transformer_encoder.num_layers
    num_heads = model.transformer_encoder.layers[0].self_attn.num_heads
    norm_first = model.transformer_encoder.layers[0].norm_first
    with torch.no_grad():
        for i in range(num_layers):
            # compute attention of layer i
            h = src.clone()
            if norm_first:
                h = model.transformer_encoder.layers[i].norm1(h)
            attn = model.transformer_encoder.layers[i].self_attn(h, h, h,attn_mask=None,key_padding_mask=None,need_weights=True)[1]
            attention_maps.append(attn)
            # forward of layer i
            src = model.transformer_encoder.layers[i](src,src_mask=None,src_key_padding_mask=None)
    return attention_maps, pred