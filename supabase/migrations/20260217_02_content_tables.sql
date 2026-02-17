-- Migration 2: Content Tables (FAQs and Products)
-- Created: 2026-02-17
-- Description: FAQ entries and product snapshots from Salla

-- ============================================
-- 1. FAQ_ENTRIES TABLE
-- ============================================

CREATE TABLE faq_entries (
  -- Primary Key
  faq_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Key
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,

  -- FAQ Data
  category VARCHAR(50) NOT NULL CHECK (category IN ('shipping', 'payment', 'returns', 'products', 'general')),
  question_ar TEXT NOT NULL CHECK (char_length(question_ar) > 0),
  answer_ar TEXT NOT NULL CHECK (char_length(answer_ar) > 0),

  -- Metadata
  is_active BOOLEAN NOT NULL DEFAULT true,
  usage_count INTEGER NOT NULL DEFAULT 0 CHECK (usage_count >= 0),
  last_used_at TIMESTAMPTZ,

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE faq_entries IS 'Merchant-provided FAQ entries for bot responses';
COMMENT ON COLUMN faq_entries.usage_count IS 'Number of times this FAQ was used in bot responses';

-- Indexes
CREATE INDEX idx_faq_store_active ON faq_entries(store_id, is_active) WHERE is_active = true;
CREATE INDEX idx_faq_category ON faq_entries(store_id, category) WHERE is_active = true;
CREATE INDEX idx_faq_usage ON faq_entries(store_id, usage_count DESC) WHERE is_active = true;

-- Full-text search index (Arabic)
CREATE INDEX idx_faq_search ON faq_entries
  USING GIN (to_tsvector('arabic', question_ar || ' ' || answer_ar));

-- ============================================
-- 2. PRODUCT_SNAPSHOTS TABLE
-- ============================================

CREATE TABLE product_snapshots (
  -- Primary Key
  snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Foreign Key
  store_id UUID NOT NULL REFERENCES stores(store_id) ON DELETE CASCADE,

  -- Product Data (from Salla API)
  salla_product_id VARCHAR(255) NOT NULL,
  name_ar VARCHAR(500) NOT NULL,
  description_ar TEXT,
  price DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
  currency VARCHAR(3) NOT NULL DEFAULT 'SAR',
  available BOOLEAN NOT NULL DEFAULT true,
  stock_quantity INTEGER CHECK (stock_quantity >= 0),
  image_url TEXT,
  category_ar VARCHAR(255),
  sku VARCHAR(100),

  -- Snapshot Metadata
  snapshot_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  is_latest BOOLEAN NOT NULL DEFAULT true,

  -- Audit
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Constraints
  UNIQUE(store_id, salla_product_id, snapshot_timestamp)
);

COMMENT ON TABLE product_snapshots IS 'Periodic snapshots of Salla products';
COMMENT ON COLUMN product_snapshots.is_latest IS 'True for the most recent snapshot per product';
COMMENT ON COLUMN product_snapshots.snapshot_timestamp IS 'When this snapshot was taken from Salla API';

-- Indexes
CREATE INDEX idx_products_store_latest ON product_snapshots(store_id, is_latest)
  WHERE is_latest = true;

CREATE INDEX idx_products_salla_id ON product_snapshots(store_id, salla_product_id)
  WHERE is_latest = true;

CREATE INDEX idx_products_available ON product_snapshots(store_id, available)
  WHERE is_latest = true AND available = true;

CREATE INDEX idx_products_snapshot_time ON product_snapshots(snapshot_timestamp DESC);

-- Full-text search index (Arabic)
CREATE INDEX idx_products_search ON product_snapshots
  USING GIN (to_tsvector('arabic', name_ar || ' ' || COALESCE(description_ar, '')))
  WHERE is_latest = true;

-- ============================================
-- 3. TRIGGERS
-- ============================================

CREATE TRIGGER update_faq_entries_updated_at
  BEFORE UPDATE ON faq_entries
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 4. FUNCTIONS
-- ============================================

-- Function: Mark old snapshots as not latest when new sync happens
CREATE OR REPLACE FUNCTION mark_old_snapshots_not_latest()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.is_latest = true THEN
    -- Set all other snapshots for this product to is_latest = false
    UPDATE product_snapshots
    SET is_latest = false
    WHERE store_id = NEW.store_id
      AND salla_product_id = NEW.salla_product_id
      AND snapshot_id != NEW.snapshot_id
      AND is_latest = true;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_mark_old_snapshots
  AFTER INSERT ON product_snapshots
  FOR EACH ROW
  EXECUTE FUNCTION mark_old_snapshots_not_latest();

COMMENT ON FUNCTION mark_old_snapshots_not_latest IS 'Automatically marks old product snapshots as not latest when new snapshot inserted';

-- Function: Increment FAQ usage count
CREATE OR REPLACE FUNCTION increment_faq_usage(faq_entry_id UUID)
RETURNS VOID AS $$
BEGIN
  UPDATE faq_entries
  SET
    usage_count = usage_count + 1,
    last_used_at = NOW()
  WHERE faq_id = faq_entry_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION increment_faq_usage IS 'Increments usage counter when FAQ is used in bot response';
