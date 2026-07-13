/**
 * This is separate branch which implement only simple embedding-pipeline.
 */
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { CohereEmbeddings } from "@langchain/cohere";
import { PineconeStore } from '@langchain/pinecone';
import { Pinecone as PineconeClient } from '@pinecone-database/pinecone'

import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import * as dotenv from 'dotenv';

dotenv.config()

export async function docEmbedding(url: string) {
    const loader = new CheerioWebBaseLoader(url)
    const parseDocs = await loader.load()

    const DocsWithMeta = parseDocs.map((doc) => {
        doc.metadata.source = url;
        return doc;
    })

    const docs = DocsWithMeta.flat();

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

    console.log("finished Embedding .... docs added", )    
}

await docEmbedding("https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/")