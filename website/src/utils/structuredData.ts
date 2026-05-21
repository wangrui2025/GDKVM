export function getScholarlyArticleSchema(data: {
  title: string;
  authors: string[];
  description: string;
  url: string;
  image: string;
  datePublished: string;
  venue: string;
  pdfUrl?: string;
  codeUrl?: string;
}) {
  return {
    '@context': 'https://schema.org',
    '@type': 'ScholarlyArticle',
    headline: data.title,
    name: data.title,
    author: data.authors.map((name) => ({
      '@type': 'Person',
      name,
    })),
    description: data.description,
    url: data.url,
    image: data.image,
    datePublished: data.datePublished,
    isPartOf: {
      '@type': 'PublicationEvent',
      name: data.venue,
    },
    ...(data.pdfUrl && { associatedMedia: { '@type': 'MediaObject', contentUrl: data.pdfUrl } }),
    ...(data.codeUrl && { codeRepository: data.codeUrl }),
  };
}
