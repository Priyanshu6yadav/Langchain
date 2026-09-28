from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import pipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

model1 = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
pipe = pipeline(
    task='text-generation',
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    max_new_tokens=250,
    return_full_text=False,
    device='mps'  # GPU acceleration on Mac Apple Silicon
)

llm = HuggingFacePipeline(pipeline=pipe)
model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = "Generate short and simple notes from the following text \n {text}",
    input_variables = ['text']
)
prompt2 = PromptTemplate(
    template = "Generate 5 short question and answers from the following text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single documents \n notes -> {notes} and quiz -> {quiz}',
    input_variables = ['notes','quiz']
)

parser = StrOutputParser()

 # Part 1 making  paraller chain
paraller_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser, # chain 1
    'quiz': prompt2 | model2 | parser # 2nd chain both work prallely
})
text = """LinearRegression
class sklearn.linear_model.LinearRegression(*, fit_intercept=True, copy_X=True, tol=1e-06, n_jobs=None, positive=False)[source]
Ordinary least squares Linear Regression.
LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.
Parameters:
fit_interceptbool, default=True
Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
copy_Xbool, default=True
If True, X will be copied; else, it may be overwritten.
tolfloat, default=1e-6
The precision of the solution (coef_) is determined by tol which specifies the convergence criterion of the underlying solver. tol is set as atol and btol of scipy.sparse.linalg.lsqr when fitting on sparse training data. tol is set as cond of scipy.linalg.lstsq when fitting on dense training data.
Added in version 1.7.
Changed in version 1.9: Now supported on dense data, interpreted as the cond parameter.
n_jobsint, default=None
The number of jobs to use for the computation. This will only provide speedup in case of sufficiently large problems, that is if firstly n_targets > 1 and secondly X is sparse or if positive is set to True. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
positivebool, default=False
When set to True, forces the coefficients to be positive. This option is only supported for dense arrays.
For a comparison between a linear regression model with positive constraints on the regression coefficients and a linear regression without such constraints, see Non-negative least squares.
Added in version 0.24.
Attributes:
coef_array of shape (n_features, ) or (n_targets, n_features)
Estimated coefficients for the linear regression problem. If multiple targets are passed during the fit (y 2D), this is a 2D array of shape (n_targets, n_features), while if only one target is passed, this is a 1D array of length n_features.
rank_int
Rank of matrix X. Only available when X is dense.
singular_array of shape (min(X, y),)
Singular values of X. Only available when X is dense.
intercept_float or array of shape (n_targets,)
Independent term in the linear model. Set to 0.0 if fit_intercept = False.
n_features_in_int
Number of features seen during fit.
Added in version 0.24.
feature_names_in_ndarray of shape (n_features_in_,)
Names of features seen during fit. Defined only when X has feature names that are all strings."""
# part 2 merge both part of the chain
merge_chian = prompt3 | model1 | parser
chain = paraller_chain | merge_chian

result = chain.invoke({'text':text})
print(result)
chain.get_graph().print_ascii()

"""          +---------------------------+            
          | Parallel<notes,quiz>Input |            
          +---------------------------+            
                ***             ***                
              **                   **              
            **                       **            
+----------------+              +----------------+ 
| PromptTemplate |              | PromptTemplate | 
+----------------+              +----------------+ 
          *                             *          
          *                             *          
          *                             *          
    +----------+               +-----------------+ 
    | ChatGroq |               | ChatHuggingFace | 
    +----------+               +-----------------+ 
          *                             *          
          *                             *          
          *                             *          
+-----------------+            +-----------------+ 
| StrOutputParser |            | StrOutputParser | 
+-----------------+            +-----------------+ 
                ***             ***                
                   **         **                   
                     **     **                     
          +----------------------------+           
          | Parallel<notes,quiz>Output |           
          +----------------------------+           
                         *                         
                         *                         
                         *                         
                +----------------+                 
                | PromptTemplate |                 
                +----------------+                 
                         *                         
                         *                         
                         *                         
                   +----------+                    
                   | ChatGroq |                    
                   +----------+                    
                         *                         
                         *                         
                         *                         
                +-----------------+                
                | StrOutputParser |                
                +-----------------+                
                         *                         
                         *                         
                         *                         
            +-----------------------+              
            | StrOutputParserOutput |              
            +-----------------------+ """