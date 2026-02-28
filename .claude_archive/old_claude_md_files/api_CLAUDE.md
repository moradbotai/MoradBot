# API Worker — CLAUDE.md

## التقنية
Cloudflare Worker بـ TypeScript + Hono Framework

## Endpoints المطلوبة
| Endpoint | Method | الوظيفة |
|----------|--------|---------|
| /auth/salla/start | GET | بدء OAuth |
| /auth/salla/callback | GET | استقبال code |
| /auth/salla/refresh | POST | تجديد token |
| /api/chat | POST | استقبال رسائل Widget |
| /api/faq | GET/POST | قراءة/تحديث FAQ |
| /api/stats | GET | إحصاءات Dashboard |
| /api/tickets | GET | قائمة المحادثات |
| /api/escalations | GET/PATCH | التصعيدات |

## قواعد هذا الـ Worker
- كل request يجب أن يحمل store_id صحيح
- Rate Limiting على كل endpoint
- Audit Log لكل عملية حساسة
- Timeout: 30 ثانية max
