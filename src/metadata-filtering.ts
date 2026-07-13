import { Document } from '@langchain/core/documents'
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { CohereEmbeddings } from "@langchain/cohere";
import { PineconeStore } from '@langchain/pinecone';
import { Pinecone as PineconeClient } from '@pinecone-database/pinecone'

import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import * as dotenv from 'dotenv';

dotenv.config()

export async function docEmbedding() {
    // const loader = new CheerioWebBaseLoader(url)
    // const parseDocs = await loader.load()

    // const DocsWithMeta = parseDocs.map((doc) => {
    //     doc.metadata.source = url;
    //     return doc;
    // })

    const docs = [
        new Document({
            pageContent: "Ben is a programmer",
            metadata: {
                userId: "doc1",
            }
        }),
        new Document({
            pageContent: "John is a driver",
            metadata: {
                userId: "doc2",
            }
        }),
        new Document({
            pageContent: "John is a doctor",
            metadata: {
                userId: "doc2",
            }
        })
    ]

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    })

    const allSplits = await textSplitter.splitDocuments(docs);

    const embeddings = new CohereEmbeddings({
        model: "embed-english-v3.0",
        apiKey: process.env.COHERE_API_KEY as string
    })

    const pinecone = new PineconeClient({
        apiKey: process.env.PINECONE_API_KEY as string,
    })
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX as string)

    const vectorStore = new PineconeStore(embeddings, { pineconeIndex, maxConcurrency: 5 })
    await vectorStore.addDocuments(allSplits)

    console.log("finished Embedding .... docs added",)
}

// await docEmbedding()

async function queryVectorDB(query : string) {

    const embeddings = new CohereEmbeddings({
        model: "embed-english-v3.0",
        apiKey: process.env.COHERE_API_KEY as string
    })

    const pinecone = new PineconeClient({
        apiKey: process.env.PINECONE_API_KEY as string
    })

    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX as string)

    const vectorStore = new PineconeStore(embeddings, { pineconeIndex, maxConcurrency: 5 })

    const result = await vectorStore.similaritySearch(query, 10, {
        userId : "doc2"
    })

    return result
}

const result = await queryVectorDB("doctor")

console.log(result)