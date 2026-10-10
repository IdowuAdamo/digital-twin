variable "project_name" {
  description = "Name prefix for all resources"
  type        = string
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment name (dev, test, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "test", "prod"], var.environment)
    error_message = "Environment must be one of: dev, test, prod."
  }
}

variable "bedrock_model_id" {
  description = "Bedrock model ID"
  type        = string
  default     = "eu.amazon.nova-lite-v1:0"
}

variable "bedrock_region" {
  description = "AWS region for Bedrock API calls (can differ from deployment region)"
  type        = string
  default     = "eu-north-1"
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 60
}

variable "api_throttle_burst_limit" {
  description = "API Gateway throttle burst limit"
  type        = number
  default     = 10
}

variable "api_throttle_rate_limit" {
  description = "API Gateway throttle rate limit"
  type        = number
  default     = 5
}

variable "use_custom_domain" {
  description = "Attach a custom domain to CloudFront"
  type        = bool
  default     = false
}

variable "root_domain" {
  description = "Apex domain name, e.g. mydomain.com"
  type        = string
  default     = ""
}
variable "ai_provider" {
  description = "AI provider to use: 'auto' (try Bedrock, fall back to OpenAI), 'bedrock', or 'openai'"
  type        = string
  default     = "auto"
  validation {
    condition     = contains(["auto", "bedrock", "openai"], var.ai_provider)
    error_message = "ai_provider must be one of: auto, bedrock, openai."
  }
}

variable "openai_api_key" {
  description = "OpenAI API key used as fallback when Bedrock is unavailable"
  type        = string
  sensitive   = true
  default     = ""
}

variable "openai_model" {
  description = "OpenAI model to use when falling back from Bedrock"
  type        = string
  default     = "gpt-4o-mini"
}
