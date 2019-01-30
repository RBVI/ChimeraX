class TutorialGUI(HtmlToolInstance):

    # previously implemented parts of the class here...

    def take_snapshot(self, session, flags):
        # For now, the 'flags' argument can be ignored.  In the
        # future, it will be used to distnguish between saving
        # for inclusion in a session vs. inclusion in a scene
        #
        # take_snapshot can actually return any type of data
        # it wants, but a dictionary is usually preferred because
        # it is easy to add to if the tool is later enhanced or
        # modified.  Also, the data returned has to consist of
        # builtin Python types (including numpy/tinyarray
        # types) and/or class instances that derive from State.

        return {
            # The 'version' key not strictly necessary here,
            # but will simplify coding the restore_snapshot
            # method in the future if the format of the
            # data dictionary is changed
            'version': 1,
            'prev_command': self.prev_command
        }

    @classmethod
    def restore_snapshot(class_obj, session, data):
        # This could also be coded as an @staticmethod, in which
        # case you would have to use the actual class name in 
        # lieu of 'class_obj' below
        #
        # 'data' is what take_snaphot returned.  At this time,
        # we have no need for the 'version' key of 'data'
        inst = class_obj(session, "Tutorial GUI")
        inst.prev_command = data['prev_command']
        return inst
