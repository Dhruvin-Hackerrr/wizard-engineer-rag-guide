import { CohereEmbeddings } from "@langchain/cohere";
import { PineconeStore } from '@langchain/pinecone';
import { Pinecone as PineconeClient } from '@pinecone-database/pinecone'

import * as dotenv from 'dotenv';

dotenv.config()

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

const result = await queryVectorDB('Types of prompt engineering')
console.log('Query Result :', result)