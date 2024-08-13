const express = require("express");
const NextFunction = express.NextFunction;
const req = express.Request;
const res = express.Response;
const TestService = require("../services/service.js");
class TestController {
  async trainModel(req, res, NextFunction) {}
}
module.exports = new TestController();
