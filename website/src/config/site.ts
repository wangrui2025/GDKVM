/**
 * Site-wide configuration constants. Centralized so URL changes only need
 * to be made in one place (e.g. paper repository renames, CDN tag bumps).
 *
 * CDN_BASE — jsDelivr tag for academic asset library. Bump a new semver tag
 * in `mykcs/academic` and update this constant to pick up new paper images.
 * See /Users/myk/Repo/webs/academic/CONTEXT.md "学术资产库化状态" for SOP.
 *
 * All other URLs are static (no need to bump a CDN tag).
 */
export const CDN_BASE = 'https://cdn.jsdelivr.net/gh/mykcs/academic@v1.1.0';

export const SITE_CONFIG = {
  /** GitHub repository for code releases */
  codeUrl: 'https://github.com/wangrui2025/GDKVM',
  /** arXiv abstract page */
  arxivUrl: 'https://arxiv.org/abs/2512.10252',
  /** CVF open access paper page */
  cvfPaperUrl:
    'https://openaccess.thecvf.com/content/ICCV2025/html/Wang_GDKVM_Echocardiography_Video_Segmentation_via_Spatiotemporal_Key-Value_Memory_with_Gated_ICCV_2025_paper.html',
  /** CVF conference poster image (direct PNG) */
  posterUrl:
    'https://iccv.thecvf.com/media/PosterPDFs/ICCV%202025/2134.png?t=1759901950.5662367',
  /** First-public publication date (ISO 8601) */
  datePublished: '2025-10-01',
  /** Venue name for structured data (schema.org PublicationEvent) */
  venue: 'ICCV 2025',
} as const;
