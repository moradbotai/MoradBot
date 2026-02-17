/**
 * Database TypeScript Types
 * Auto-generated types for Supabase schema
 * Generated: 2026-02-17
 */

export type Json =
	| string
	| number
	| boolean
	| null
	| { [key: string]: Json | undefined }
	| Json[];

export interface Database {
	public: {
		Tables: {
			plans: {
				Row: {
					plan_id: string;
					plan_name: "basic" | "mid" | "premium";
					plan_name_ar: string;
					bot_reply_limit: number;
					sync_frequency_hours: 1 | 6 | 24;
					price_monthly_sar: number;
					features: Json;
					is_active: boolean;
					created_at: string;
					updated_at: string;
				};
				Insert: {
					plan_id?: string;
					plan_name: "basic" | "mid" | "premium";
					plan_name_ar: string;
					bot_reply_limit: number;
					sync_frequency_hours: 1 | 6 | 24;
					price_monthly_sar: number;
					features?: Json;
					is_active?: boolean;
					created_at?: string;
					updated_at?: string;
				};
				Update: {
					plan_id?: string;
					plan_name?: "basic" | "mid" | "premium";
					plan_name_ar?: string;
					bot_reply_limit?: number;
					sync_frequency_hours?: 1 | 6 | 24;
					price_monthly_sar?: number;
					features?: Json;
					is_active?: boolean;
					created_at?: string;
					updated_at?: string;
				};
			};
			stores: {
				Row: {
					store_id: string;
					salla_merchant_id: string;
					salla_access_token: string | null;
					salla_refresh_token: string | null;
					salla_token_expires_at: string | null;
					store_name_ar: string;
					store_url: string;
					contact_email: string | null;
					contact_phone: string | null;
					bot_enabled: boolean;
					widget_settings: Json;
					created_at: string;
					updated_at: string;
					deleted_at: string | null;
				};
				Insert: {
					store_id?: string;
					salla_merchant_id: string;
					salla_access_token?: string | null;
					salla_refresh_token?: string | null;
					salla_token_expires_at?: string | null;
					store_name_ar: string;
					store_url: string;
					contact_email?: string | null;
					contact_phone?: string | null;
					bot_enabled?: boolean;
					widget_settings?: Json;
					created_at?: string;
					updated_at?: string;
					deleted_at?: string | null;
				};
				Update: {
					store_id?: string;
					salla_merchant_id?: string;
					salla_access_token?: string | null;
					salla_refresh_token?: string | null;
					salla_token_expires_at?: string | null;
					store_name_ar?: string;
					store_url?: string;
					contact_email?: string | null;
					contact_phone?: string | null;
					bot_enabled?: boolean;
					widget_settings?: Json;
					created_at?: string;
					updated_at?: string;
					deleted_at?: string | null;
				};
			};
			subscriptions: {
				Row: {
					subscription_id: string;
					store_id: string;
					plan_id: string;
					status: "active" | "cancelled" | "suspended";
					started_at: string;
					ends_at: string | null;
					current_cycle_start: string;
					current_cycle_end: string;
					current_cycle_usage: number;
					created_at: string;
					updated_at: string;
				};
				Insert: {
					subscription_id?: string;
					store_id: string;
					plan_id: string;
					status?: "active" | "cancelled" | "suspended";
					started_at?: string;
					ends_at?: string | null;
					current_cycle_start?: string;
					current_cycle_end?: string;
					current_cycle_usage?: number;
					created_at?: string;
					updated_at?: string;
				};
				Update: {
					subscription_id?: string;
					store_id?: string;
					plan_id?: string;
					status?: "active" | "cancelled" | "suspended";
					started_at?: string;
					ends_at?: string | null;
					current_cycle_start?: string;
					current_cycle_end?: string;
					current_cycle_usage?: number;
					created_at?: string;
					updated_at?: string;
				};
			};
			faq_entries: {
				Row: {
					faq_id: string;
					store_id: string;
					category: "shipping" | "payment" | "returns" | "products" | "general";
					question_ar: string;
					answer_ar: string;
					is_active: boolean;
					usage_count: number;
					last_used_at: string | null;
					created_at: string;
					updated_at: string;
				};
				Insert: {
					faq_id?: string;
					store_id: string;
					category: "shipping" | "payment" | "returns" | "products" | "general";
					question_ar: string;
					answer_ar: string;
					is_active?: boolean;
					usage_count?: number;
					last_used_at?: string | null;
					created_at?: string;
					updated_at?: string;
				};
				Update: {
					faq_id?: string;
					store_id?: string;
					category?: "shipping" | "payment" | "returns" | "products" | "general";
					question_ar?: string;
					answer_ar?: string;
					is_active?: boolean;
					usage_count?: number;
					last_used_at?: string | null;
					created_at?: string;
					updated_at?: string;
				};
			};
			product_snapshots: {
				Row: {
					snapshot_id: string;
					store_id: string;
					salla_product_id: string;
					name_ar: string;
					description_ar: string | null;
					price: number;
					currency: string;
					available: boolean;
					stock_quantity: number | null;
					image_url: string | null;
					category_ar: string | null;
					sku: string | null;
					snapshot_timestamp: string;
					is_latest: boolean;
					created_at: string;
				};
				Insert: {
					snapshot_id?: string;
					store_id: string;
					salla_product_id: string;
					name_ar: string;
					description_ar?: string | null;
					price: number;
					currency?: string;
					available?: boolean;
					stock_quantity?: number | null;
					image_url?: string | null;
					category_ar?: string | null;
					sku?: string | null;
					snapshot_timestamp?: string;
					is_latest?: boolean;
					created_at?: string;
				};
				Update: {
					snapshot_id?: string;
					store_id?: string;
					salla_product_id?: string;
					name_ar?: string;
					description_ar?: string | null;
					price?: number;
					currency?: string;
					available?: boolean;
					stock_quantity?: number | null;
					image_url?: string | null;
					category_ar?: string | null;
					sku?: string | null;
					snapshot_timestamp?: string;
					is_latest?: boolean;
					created_at?: string;
				};
			};
			visitor_sessions: {
				Row: {
					visitor_id: string;
					store_id: string;
					session_cookie: string;
					consent_given: boolean;
					consent_given_at: string | null;
					email_encrypted: string | null;
					phone_encrypted: string | null;
					name_encrypted: string | null;
					first_visit_at: string;
					last_visit_at: string;
					visit_count: number;
					user_agent: string | null;
					ip_address: string | null;
					created_at: string;
					updated_at: string;
				};
				Insert: {
					visitor_id?: string;
					store_id: string;
					session_cookie: string;
					consent_given?: boolean;
					consent_given_at?: string | null;
					email_encrypted?: string | null;
					phone_encrypted?: string | null;
					name_encrypted?: string | null;
					first_visit_at?: string;
					last_visit_at?: string;
					visit_count?: number;
					user_agent?: string | null;
					ip_address?: string | null;
					created_at?: string;
					updated_at?: string;
				};
				Update: {
					visitor_id?: string;
					store_id?: string;
					session_cookie?: string;
					consent_given?: boolean;
					consent_given_at?: string | null;
					email_encrypted?: string | null;
					phone_encrypted?: string | null;
					name_encrypted?: string | null;
					first_visit_at?: string;
					last_visit_at?: string;
					visit_count?: number;
					user_agent?: string | null;
					ip_address?: string | null;
					created_at?: string;
					updated_at?: string;
				};
			};
			tickets: {
				Row: {
					ticket_id: string;
					store_id: string;
					visitor_id: string;
					status: "open" | "resolved" | "escalated" | "closed";
					resolution_type:
						| "bot_answered"
						| "escalated"
						| "auto_closed"
						| "merchant_closed"
						| null;
					initial_question: string | null;
					category: string | null;
					clarification_count: number;
					opened_at: string;
					resolved_at: string | null;
					escalated_at: string | null;
					closed_at: string | null;
					created_at: string;
					updated_at: string;
				};
				Insert: {
					ticket_id?: string;
					store_id: string;
					visitor_id: string;
					status?: "open" | "resolved" | "escalated" | "closed";
					resolution_type?:
						| "bot_answered"
						| "escalated"
						| "auto_closed"
						| "merchant_closed"
						| null;
					initial_question?: string | null;
					category?: string | null;
					clarification_count?: number;
					opened_at?: string;
					resolved_at?: string | null;
					escalated_at?: string | null;
					closed_at?: string | null;
					created_at?: string;
					updated_at?: string;
				};
				Update: {
					ticket_id?: string;
					store_id?: string;
					visitor_id?: string;
					status?: "open" | "resolved" | "escalated" | "closed";
					resolution_type?:
						| "bot_answered"
						| "escalated"
						| "auto_closed"
						| "merchant_closed"
						| null;
					initial_question?: string | null;
					category?: string | null;
					clarification_count?: number;
					opened_at?: string;
					resolved_at?: string | null;
					escalated_at?: string | null;
					closed_at?: string | null;
					created_at?: string;
					updated_at?: string;
				};
			};
			messages: {
				Row: {
					message_id: string;
					ticket_id: string;
					store_id: string;
					sender_type: "visitor" | "bot" | "merchant";
					content_ar: string;
					is_clarification_request: boolean;
					includes_dynamic_data: boolean;
					model_used: string | null;
					tokens_used: number | null;
					response_time_ms: number | null;
					created_at: string;
				};
				Insert: {
					message_id?: string;
					ticket_id: string;
					store_id: string;
					sender_type: "visitor" | "bot" | "merchant";
					content_ar: string;
					is_clarification_request?: boolean;
					includes_dynamic_data?: boolean;
					model_used?: string | null;
					tokens_used?: number | null;
					response_time_ms?: number | null;
					created_at?: string;
				};
				Update: {
					message_id?: string;
					ticket_id?: string;
					store_id?: string;
					sender_type?: "visitor" | "bot" | "merchant";
					content_ar?: string;
					is_clarification_request?: boolean;
					includes_dynamic_data?: boolean;
					model_used?: string | null;
					tokens_used?: number | null;
					response_time_ms?: number | null;
					created_at?: string;
				};
			};
			escalations: {
				Row: {
					escalation_id: string;
					ticket_id: string;
					store_id: string;
					visitor_id: string;
					reason:
						| "failed_clarification"
						| "unsupported_request"
						| "manual_request"
						| "error";
					problem_description: string;
					contact_method: "email" | "phone";
					contact_value_encrypted: string;
					order_number: string | null;
					status: "pending" | "in_progress" | "resolved" | "closed";
					resolved_by: "merchant" | "system" | null;
					resolution_notes: string | null;
					escalated_at: string;
					resolved_at: string | null;
					created_at: string;
					updated_at: string;
				};
				Insert: {
					escalation_id?: string;
					ticket_id: string;
					store_id: string;
					visitor_id: string;
					reason:
						| "failed_clarification"
						| "unsupported_request"
						| "manual_request"
						| "error";
					problem_description: string;
					contact_method: "email" | "phone";
					contact_value_encrypted: string;
					order_number?: string | null;
					status?: "pending" | "in_progress" | "resolved" | "closed";
					resolved_by?: "merchant" | "system" | null;
					resolution_notes?: string | null;
					escalated_at?: string;
					resolved_at?: string | null;
					created_at?: string;
					updated_at?: string;
				};
				Update: {
					escalation_id?: string;
					ticket_id?: string;
					store_id?: string;
					visitor_id?: string;
					reason?:
						| "failed_clarification"
						| "unsupported_request"
						| "manual_request"
						| "error";
					problem_description?: string;
					contact_method?: "email" | "phone";
					contact_value_encrypted?: string;
					order_number?: string | null;
					status?: "pending" | "in_progress" | "resolved" | "closed";
					resolved_by?: "merchant" | "system" | null;
					resolution_notes?: string | null;
					escalated_at?: string;
					resolved_at?: string | null;
					created_at?: string;
					updated_at?: string;
				};
			};
			usage_events: {
				Row: {
					event_id: string;
					store_id: string;
					subscription_id: string | null;
					ticket_id: string | null;
					message_id: string | null;
					event_type: "bot_reply" | "clarification" | "escalation";
					billing_cycle_start: string;
					billing_cycle_end: string;
					tokens_used: number | null;
					estimated_cost_usd: number | null;
					model_used: string | null;
					created_at: string;
				};
				Insert: {
					event_id?: string;
					store_id: string;
					subscription_id?: string | null;
					ticket_id?: string | null;
					message_id?: string | null;
					event_type: "bot_reply" | "clarification" | "escalation";
					billing_cycle_start: string;
					billing_cycle_end: string;
					tokens_used?: number | null;
					estimated_cost_usd?: number | null;
					model_used?: string | null;
					created_at?: string;
				};
				Update: {
					event_id?: string;
					store_id?: string;
					subscription_id?: string | null;
					ticket_id?: string | null;
					message_id?: string | null;
					event_type?: "bot_reply" | "clarification" | "escalation";
					billing_cycle_start?: string;
					billing_cycle_end?: string;
					tokens_used?: number | null;
					estimated_cost_usd?: number | null;
					model_used?: string | null;
					created_at?: string;
				};
			};
			consent_logs: {
				Row: {
					consent_id: string;
					store_id: string;
					visitor_id: string;
					ticket_id: string | null;
					consent_type: "personal_data_storage" | "persistent_memory";
					consent_given: boolean;
					consent_method: "chat_checkbox" | "explicit_message" | "system_default";
					ip_address: string | null;
					user_agent: string | null;
					created_at: string;
				};
				Insert: {
					consent_id?: string;
					store_id: string;
					visitor_id: string;
					ticket_id?: string | null;
					consent_type: "personal_data_storage" | "persistent_memory";
					consent_given: boolean;
					consent_method: "chat_checkbox" | "explicit_message" | "system_default";
					ip_address?: string | null;
					user_agent?: string | null;
					created_at?: string;
				};
				Update: {
					// Immutable table - no updates allowed
					consent_id?: never;
				};
			};
			audit_logs: {
				Row: {
					audit_id: string;
					store_id: string | null;
					actor_type: "merchant" | "staff" | "system" | "api";
					actor_id: string | null;
					action: string;
					resource_type: string | null;
					resource_id: string | null;
					ip_address: string | null;
					user_agent: string | null;
					metadata: Json;
					created_at: string;
				};
				Insert: {
					audit_id?: string;
					store_id?: string | null;
					actor_type: "merchant" | "staff" | "system" | "api";
					actor_id?: string | null;
					action: string;
					resource_type?: string | null;
					resource_id?: string | null;
					ip_address?: string | null;
					user_agent?: string | null;
					metadata?: Json;
					created_at?: string;
				};
				Update: {
					// Immutable table - no updates allowed
					audit_id?: never;
				};
			};
		};
		Views: {
			v_active_subscriptions: {
				Row: {
					subscription_id: string;
					store_id: string;
					store_name_ar: string;
					plan_name: "basic" | "mid" | "premium";
					bot_reply_limit: number;
					current_cycle_usage: number;
					remaining_usage: number;
					usage_percentage: number;
				};
			};
			v_pending_escalations: {
				Row: {
					escalation_id: string;
					store_id: string;
					store_name_ar: string;
					ticket_id: string;
					problem_description: string;
					contact_method: "email" | "phone";
					order_number: string | null;
					escalated_at: string;
					message_count: number;
				};
			};
		};
		Functions: {
			increment_faq_usage: {
				Args: { faq_entry_id: string };
				Returns: void;
			};
			log_audit_event: {
				Args: {
					p_store_id: string | null;
					p_actor_type: "merchant" | "staff" | "system" | "api";
					p_actor_id: string;
					p_action: string;
					p_resource_type?: string | null;
					p_resource_id?: string | null;
					p_metadata?: Json;
				};
				Returns: string;
			};
			log_consent_event: {
				Args: {
					p_store_id: string;
					p_visitor_id: string;
					p_ticket_id: string | null;
					p_consent_type: "personal_data_storage" | "persistent_memory";
					p_consent_given: boolean;
					p_consent_method: "chat_checkbox" | "explicit_message" | "system_default";
					p_ip_address?: string | null;
					p_user_agent?: string | null;
				};
				Returns: string;
			};
		};
	};
}

// Helper types
export type Tables<T extends keyof Database["public"]["Tables"]> =
	Database["public"]["Tables"][T]["Row"];

export type Inserts<T extends keyof Database["public"]["Tables"]> =
	Database["public"]["Tables"][T]["Insert"];

export type Updates<T extends keyof Database["public"]["Tables"]> =
	Database["public"]["Tables"][T]["Update"];

export type Views<T extends keyof Database["public"]["Views"]> =
	Database["public"]["Views"][T]["Row"];

// Specific table types for convenience
export type Plan = Tables<"plans">;
export type Store = Tables<"stores">;
export type Subscription = Tables<"subscriptions">;
export type FAQEntry = Tables<"faq_entries">;
export type ProductSnapshot = Tables<"product_snapshots">;
export type VisitorSession = Tables<"visitor_sessions">;
export type Ticket = Tables<"tickets">;
export type Message = Tables<"messages">;
export type Escalation = Tables<"escalations">;
export type UsageEvent = Tables<"usage_events">;
export type ConsentLog = Tables<"consent_logs">;
export type AuditLog = Tables<"audit_logs">;

// View types
export type ActiveSubscription = Views<"v_active_subscriptions">;
export type PendingEscalation = Views<"v_pending_escalations">;
