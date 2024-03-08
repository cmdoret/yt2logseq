import sys

from langchain.chains import LLMChain, MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_openai import ChatOpenAI

from yt2logseq.prompts import MAP_TMPL, REDUCE_TMPL
from yt2logseq.srt import StampedSRTLoader

llm = ChatOpenAI(temperature=0)

def make_map_chain() -> LLMChain:
    map_prompt = PromptTemplate.from_template(MAP_TMPL)
    return LLMChain(llm=llm, prompt=map_prompt)

def make_reduce_chain() -> ReduceDocumentsChain:
    reduce_prompt = PromptTemplate.from_template(REDUCE_TMPL)
    chain = LLMChain(llm=llm, prompt=reduce_prompt)
    # combines list of docs into a single string and pass to a LLMChain
    combine_chain = StuffDocumentsChain(
        llm_chain=chain, document_variable_name="docs"
    )
    # Combines and iteratively reduces the mapped documents
    return ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )

def make_mapreduce_chain() -> MapReduceDocumentsChain:
    map_chain = make_map_chain()
    reduce_chain = make_reduce_chain()
    return MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

def summarize_subtitles(subs: str) -> str:
    loader = StampedSRTLoader(file_path="", data=subs)
    docs = loader.load()
    mapreduce_chain = make_mapreduce_chain()
    result = mapreduce_chain.invoke({'input_documents': docs})
    return result['output_text']

if __name__ == '__main__':
    sub = open(sys.argv[1]).read()
    summary = summarize_subtitles(sub)
    print(summary)

