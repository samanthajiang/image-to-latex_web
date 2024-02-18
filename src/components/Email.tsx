import { useRef, useEffect, useState } from 'react';

import emailjs from '@emailjs/browser';

const Contact = () => {
  useEffect(() => emailjs.init('a8-CHtPthMLvsJfEJ'), []);
  // const emailRef = useRef<HTMLInputElement>();
  // const msgRef = useRef<HTMLInputElement>();
  const emailRef = useRef<HTMLInputElement>(null);
  const msgRef = useRef<HTMLTextAreaElement>(null);
  const [loading, setLoading] = useState(false);
  const handleSubmit = async (e: any) => {
    e.preventDefault();
    const serviceId = 'service_442gxyv';
    const templateId = 'template_tkub0wh';
    try {
      setLoading(true);
      if (emailRef.current != null && msgRef.current != null) {
        //  TypeScript knows that ref is not null here
        await emailjs.send(serviceId, templateId, {
          from_name: emailRef.current.value,
          message: msgRef.current.value,
        });
        // eslint-disable-next-line no-alert
        alert('Email successfully sent');
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.log(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <section className="bg-white dark:bg-gray-900">
        <div className="py-8 lg:py-16 px-4 mx-auto max-w-screen-md">
          <h2 className="mb-4 text-4xl tracking-tight font-extrabold text-center text-gray-900 dark:text-white">
            Contact Us
          </h2>
          <p className="mb-8 lg:mb-16 text-center text-gray-500 dark:text-gray-400 sm:text-xl">
            Got a technical issue or have any comment? Let us know.
          </p>
          <form action="#" className="space-y-8" onSubmit={handleSubmit}>
            <div>
              <label
                htmlFor="email"
                className="block mb-2 text-lg font-medium text-gray-900 dark:text-gray-300"
              >
                Your email
              </label>
              <input
                ref={emailRef}
                type="email"
                id="email"
                className="shadow-sm bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-primary-500 focus:border-primary-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-primary-500 dark:focus:border-primary-500 dark:shadow-sm-light"
                placeholder="name@email.com"
                required
              />
            </div>
            {/* <div> */}
            {/*  <label htmlFor="subject" */}
            {/*         className="block mb-2 text-sm font-medium text-gray-900 dark:text-gray-300">Subject</label> */}
            {/*  <input type="text" id="subject" */}
            {/*         className="block p-3 w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 shadow-sm focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-primary-500 dark:focus:border-primary-500 dark:shadow-sm-light" */}
            {/*         placeholder="Let us know how we can help you" required /> */}
            {/* </div> */}
            <div className="sm:col-span-2">
              <label
                htmlFor="message"
                className="block mb-2 text-lg font-medium text-gray-900 dark:text-gray-400"
              >
                Your message
              </label>
              <textarea
                ref={msgRef}
                id="message"
                rows={6}
                className="block p-2.5 w-full text-sm text-gray-900 bg-gray-50 rounded-lg shadow-sm border border-gray-300 focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-primary-500 dark:focus:border-primary-500"
                placeholder="Leave a comment..."
              ></textarea>
            </div>
            <div className="container px-3 mx-0 min-w-full flex flex-col items-center">
              <button
                type="submit"
                disabled={loading}
                // className="w-30 flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-background bg-primary hover:bg-border hover:text-primary md:py-4 md:text-lg md:px-10"

                className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-background bg-primary hover:bg-red-200 hover:text-gray-900 md:py-4 md:text-lg md:px-10`"
              >
                Send message
              </button>
            </div>
          </form>
        </div>
      </section>
    </div>
  );
};

export default Contact;
