#!/usr/bin/python3

import unittest
import logging


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.DEBUG)
logger = logging.getLogger()

#rest_logger = logging.getLogger("cxservices.rest")
#print(rest_logger)
#rest_logger.setLevel(logging.DEBUG)


class ChimeraXTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        from cxservices.api import default_api
        self.api = default_api.DefaultApi()


class TestChimeraX(ChimeraXTestCase):

    def Xtest_sleep(self):
        result = self.api.job_id()
        job_id = result.job_id
        logger.debug("created: %s" % job_id)
        result = self.api.status(job_id)
        logger.debug("status: %s" % result.status)
        result = self.api.sleep(job_id, body={'wait_time':15})
        logger.debug("sleep: %s" % str(result))
        result = self.api.status(job_id)
        logger.debug("status: %s" % result.status)
        result = self.api.job_delete(job_id)
        logger.debug("job_delete: %s" % str(result))

    def test_blast(self):
        from cxservices.rest import ApiException
        result = self.api.job_id()
        job_id = result.job_id
        logger.debug("created: %s" % job_id)
        result = self.api.status(job_id)
        logger.debug("status: %s" % result.status)

        file_name = "upload_test"
        file_content = b'Hello world'
        result = self.api.file_post(file_content, job_id, file_name)
        logger.debug("file_post: %s" % str(result))
        result = self.api.status(job_id)
        logger.debug("status: %s" % result.status)
        result = self.api.files_list(job_id)
        logger.debug("file list: %s" % result.file_names)
        result = self.api.file_get(job_id, file_name)
        logger.debug("file content: %s" % repr(result))
        bad_file_name = file_name + ".bad"
        try:
            result = self.api.file_get(job_id, bad_file_name)
            logger.debug("file content: %s" % repr(result))
        except ApiException:
            logger.debug("no content: %s" % bad_file_name)

        seq = ("AKALIVYGSTTGNTEYTAETIARELADAGYEVDSRDAASVEAGGLFEGFDLVLLG"
               "CSTWGDDSIELQDDFIPLFDSLEETGAQGRKVACFGCGDSSYEYFCGAVDAIEEK"
               "LKNLGAEIVQDGLRIDGDPRAARDDIVGWAHDVRGAI")
        output_file = "blast.out"
        svc_opts = {"input_seq":seq, "output_file":output_file}
        result = self.api.submit(svc_opts, job_id, "blast")
        logger.debug("submit: %s" % str(result))
        import time
        while True:
            result = self.api.status(job_id)
            logger.debug("status: %s" % result.status)
            if result.status in ["complete", "failed", "deleted"]:
                break
            time.sleep(5)
        result = self.api.files_list(job_id)
        logger.debug("files: %s" % result.file_names)
        result = self.api.file_get(job_id, output_file)
        logger.debug("file content: %d bytes" % len(result))

        result = self.api.job_delete(job_id)
        logger.debug("delete: %s" % repr(result))
        try:
            result = self.api.job_delete(job_id)
            logger.debug("second delete: %s" % repr(result))
        except ApiException:
            logger.debug("second delete rejected")


if __name__ == '__main__':
    unittest.main()
