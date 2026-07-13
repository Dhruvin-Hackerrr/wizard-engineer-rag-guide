import { CohereEmbeddings } from "@langchain/cohere";
import { PineconeStore } from '@langchain/pinecone';
import { Pinecone as PineconeClient } from '@pinecone-database/pinecone';
import { PromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatCerebras } from "@langchain/cerebras";
import { Document } from "@langchain/core/documents";
import { z } from "zod";

import * as dotenv from 'dotenv';
import { queryMultiVector } from "./multi-vector-retriever.js";

dotenv.config();

const llm = new ChatCerebras({
    model: "gpt-oss-120b",
    temperature: 0,
    maxRetries: 2,
    apiKey: process.env.CEREBRAS_API_KEY as string
});

export async function queryVectorDB(query : string) {

    const embeddings = new CohereEmbeddings({
        model: "embed-english-v3.0",
        apiKey: process.env.COHERE_API_KEY as string
    })

    const pinecone = new PineconeClient({
        apiKey: process.env.PINECONE_API_KEY as string
    })

    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX as string)

    const vectorStore = new PineconeStore(embeddings, { pineconeIndex, maxConcurrency: 5 })

    const result = await vectorStore.similaritySearch(query, 1)

    return result
}

const query = "What is Chain-of-Thought (CoT)?"

const QUERY_TRANSFORMATION_PROMPT = PromptTemplate.fromTemplate(
    `
    You are an expert at query rewriting and semantic search and retrieval-augemented generation (RAG).
    
    Step back and thing about the user's underlying intent before rewriting the query.
    
    Instruction:
    1. Analyze the original question.
    2. Indentify the core goal, concepts and implied context.
    3. Generate at least 3 alternative rewritten queries that better express the same intent.
    4. Each rewritten query should be clear, specific and optimized for semantic retrieval.
    5. Do NOT add explanations or reasoning.
    
    Original Question :
    ----------- 
    {question}
    -----------
    `
)

export const GENERATE_RESPONSE_PROMPT = PromptTemplate.fromTemplate(
    `
    You are an assistant for question-answering tasks. use the following peice of retrievied contect to answer the question
    if you dont know the answer, Just say that you don't know. Use three sentences and keep the answer concise.
    
    Context : {context}
    Question : {question}
    Answer : 
    `
);

const structuredLlm = llm.withStructuredOutput(
    z.object({
        questions: z.array(z.string()).describe('array of questions for semantic search retrieval')
    }).describe('array of questions for semantic search retrieval')
)

const chain = QUERY_TRANSFORMATION_PROMPT.pipe(structuredLlm)

const generatedQuestions = await chain.invoke({ question: query })

console.log("Rewritten Questions:", generatedQuestions)

const formatDocumentAsString = (documents: Document[]) => {
    return documents.map((doc) => doc?.pageContent).join("\n\n")
}

// const docToString = formatDocumentAsString(result)
const questions = generatedQuestions?.questions

const retrieviedDocs=[]
for(const question of questions){
    const result = await queryMultiVector(question, 'doc1')
    retrieviedDocs.push(result)
}

const flattenedDocs = retrieviedDocs.flat()
const flattenedDocsToString = formatDocumentAsString(flattenedDocs)

const chain2 = GENERATE_RESPONSE_PROMPT.pipe(llm)

const aiResponse = await chain2.invoke({
    question : query,
    context : flattenedDocsToString
})

console.log('AI Response :', aiResponse)

const doc = new Document({
    pageContent: "data....",
    metadata: {
        source : "",
        url : "",
        pageNumber : ""
    }
})

function fetchDocsFromVector(userId : string, query : string){
    
}