variable "region" {
  type    = string
  default = "eu-north-1"
}

variable "name" {
  type    = string
  default = "ml"
}

variable "container_port" {
  type    = number
  default = 8000
}

variable "cpu" {
  type    = number
  default = 512
}

variable "memory" {
  type    = number
  default = 1024
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "image" {
  type        = string
  description = "ECR image URI, e.g. <acct>.dkr.ecr.eu-north-1.amazonaws.com/ml:<sha>"
}
