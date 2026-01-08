resource "aws_cloudwatch_log_group" "ml" {
  name              = "/ecs/${var.name}"
  retention_in_days = 14
}