resource "aws_ecr_repository" "ml" {
  name = "${var.name}"
}
