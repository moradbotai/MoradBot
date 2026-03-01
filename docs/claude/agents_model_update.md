# تحديث نموذج الـ Agents — Opus → Sonnet
**التاريخ:** 2026-03-01
**السبب:** توحيد النموذج وخفض التكلفة مع الحفاظ على الجودة

---

## ملخص التغييرات

| Agent | النموذج القديم | النموذج الجديد |
|-------|--------------|---------------|
| `ai-engineer.md` | opus | **sonnet** |
| `security-auditor.md` | opus | **sonnet** |

**الملفات غير المتأثرة (لم تتغير):**

| Agent | النموذج |
|-------|---------|
| `backend-architect.md` | sonnet — لم يتغير |
| `database-optimizer.md` | sonnet — لم يتغير |
| `error-detective.md` | sonnet — لم يتغير |
| `prompt-engineer.md` | sonnet — لم يتغير |
| `typescript-pro.md` | sonnet — لم يتغير |
| `api-documenter.md` | haiku — لم يتغير |
| `technical-writer.md` | haiku — لم يتغير |

---

## مسار النسخة الاحتياطية

```
/Users/mohammedaljohani/Documents/Proj/moradbot/.claude/agents_backup_2026-03-01/
```

يحتوي على النسخة الأصلية من جميع الـ 9 agents قبل التحديث.

---

## الملفات المُعدَّلة

- `/Users/mohammedaljohani/Documents/Proj/moradbot/.claude/agents/ai-engineer.md` — السطر 5
- `/Users/mohammedaljohani/Documents/Proj/moradbot/.claude/agents/security-auditor.md` — السطر 5

---

## ملاحظة

`.claude/agents/` مجلد gitignored — هذا التحديث محلي فقط ولا يُحفَظ في git.
لاستعادة النسخة القديمة:
```bash
cp -r .claude/agents_backup_2026-03-01/* .claude/agents/
```
